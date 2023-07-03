import math
import os
import time
from contextlib import nullcontext  # used for with commands
from logging import config

import torch
import torch.distributed as dist
from gpt import GPT, GPTConfigTest
from torch.nn.parallel import DistributedDataParallel as DDP

# Altered version of : https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s
# at timestamp 1hr 24mins
# Decoder only transformer, no cross attention or encoder

torch.manual_seed(1337)
ddp = False


# Training loop
# Training our model consists in repeating the following actions successively for each batch of input data at each epoch:

# Unpack input ids, attentions masks and corresponding target prices,
# Load these onto the GPU or CPU device,
# Reset the gradients of the previous training step,
# Compute the prediction (forward pass),
# Compute the gradients (backpropagation),
# Clip gradients to prevent exploding or vanishing gradient issues,
# Update the model parameters,
# Adjust the learning rate.


class hardware_setup:
    def __init__(self, ddp):
        (
            self.master_process,
            self.seed_offset,
            self.world_size,
            self.local_rank,
            self.device,
        ) = self.init_distributed(ddp)
        self.ctx, self.scaler = self.init_amp()

    def init_distributed(self, ddp):
        if ddp:
            # only works with torch.distributed.launch // torch.run
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])

            dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

            # this will make all .cuda() calls work properly
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)
            master_process = (
                rank == 0
            )  # this process will do logging, checkpointing etc.
            seed_offset = rank  # each process gets a different seed

        else:
            # if not ddp, we are running on a single gpu, and one process
            master_process = True
            seed_offset = 0
            world_size = 1
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return master_process, seed_offset, world_size, local_rank, device

    # configuring for automatic mixed precision training, see https://pytorch.org/docs/stable/amp.html
    # note: float16 data type will automatically use a GradScaler
    def init_amp(self):
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        torch.manual_seed(
            1337 + self.seed_offset
        )  # monitoring each process with a different seed
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]

        ctx = (
            nullcontext()
            if self.device() == "cpu"
            else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

        return ctx, scaler


# calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            with hw.ctx:  # for amp
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# config = BigramConfigTest(**config_args)
# model = BigramLanguageModel(config)


print("basing vocab size of Shakespeare dataset")
config_args = {}
config_args["vocab_size"] = vocab_size

# configuring process and hardware
hw = hardware_setup(ddp)
config = GPTConfigTest(**config_args)
model = GPT(config)
m = model.to(hw.device)

# create a map from chars to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# forst 90% wil be train data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate small batch of inputs x and outputs y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    x, y = x.to(hw.device), y.to(hw.device)
    return x, y


# create a PyTorch optimizer
optimizer = m.optimizer()

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[hw.local_rank])


# optimizes model for faster training using compile (torch 2.0)
def model_compile(model):
    print("compiling model...")
    time.tic()
    model = torch.compile(model)
    print(f"Compiling mode took: {time.toc()} sec")
    return model


model = model_compile(model)


# learning rate decay scheduler (cosine with warmup)
# //TODO check if this is the same as the cosine annealing function in torch
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < m.config.warmup_iters:
        return m.config.learning_rate * it / m.config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > m.config.lr_decay_iters:
        return m.config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - m.config.warmup_iters) / (
        m.config.lr_decay_iters - m.config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return m.config.min_lr + coeff * (m.config.learning_rate - m.config.min_lr)


# training loop
for iter in range(config.max_iters):
    # every onece in a while evaluate loss on train and val sets
    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']}, val loss {losses['val']}")

    # sample a batch of data
    (
        xb,
        yb,
    ) = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=hw.device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))


if ddp:
    dist.destroy_process_group()
