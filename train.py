import math
import os
import time
from contextlib import nullcontext  # used for with commands

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.gpt import GPT, GPTConfigTest

# Altered version of : https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s
# at timestamp 1hr 24mins
# Decoder only transformer, no cross attention or encoder

torch.manual_seed(1337)
#torch._dynamo.config.verbose=True
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
    def __init__(self, ddp, gradient_accum_steps):
        self.init_distributed(ddp, gradient_accum_steps)
        self.ctx, self.scaler = self.init_amp()

    def init_distributed(self, ddp, gradient_accum_steps):
        if ddp:
            # only works with torch.distributed.launch // torch.run
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])

            dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank)

            # this will make all .cuda() calls work properly
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = (
                self.rank == 0
            )  # this process will do logging, checkpointing etc.
            self.seed_offset = self.rank  # each process gets a different seed
            assert gradient_accum_steps % self.world_size == 0
            self.gradient_accum_steps = gradient_accum_steps//self.world_size

        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            self.seed_offset = 0
            self.world_size = 1
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


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
            if self.device == "cpu"
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


with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# config = BigramConfigTest(**config_args)
# model = BigramLanguageModel(config)


print("basing vocab size of Shakespeare dataset")
config_args = {}
config_args["vocab_size"] = vocab_size
config = GPTConfigTest(**config_args)

# configuring process and hardware
hw = hardware_setup(ddp, config.gradient_accum_steps) # changing gradiant accumulation based on num GUs
# https://muellerzr.github.io/blog/gradient_accumulation.html
if ddp: # if we're using mulitple GPUs
    config.gradient_accum_steps = hw.gradient_accum_steps
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
optimizer = model.create_optimizer(hw.device)


# optimizes model for faster training using compile (torch 2.0)
def model_compile(model):
    print("compiling model...")
    tic = time.time()
    model = torch.compile(model)
    toc = time.time()
    print(f"Compiling mode took: {toc-tic} sec")
    return model

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[hw.local_rank])
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
for iter in range(1, config.max_iters):
    # configure learning rate
        # determine and set the learning rate for this iteration
    lr = get_lr(iter) if config.decay_lr else config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # every onece in a while evaluate loss on train and val sets
    if iter % config.eval_interval == 0 and hw.master_process:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']}, val loss {losses['val']}")

    # sample a batch of data
    (
        xb,
        yb,
    ) = get_batch("train")

    # evaluate the loss
    with hw.ctx:
        logits, loss = model(xb, yb)
        loss = (
            loss / config.gradient_accum_steps
        )  # scale the loss to account for gradient accumulation
    
    hw.scaler.scale(loss).backward()
    # clip the gradient
    if config.grad_clip != 0.0:
        hw.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    # step the optimizer and scaler if training in fp16
    hw.scaler.step(optimizer)
    hw.scaler.update()
    optimizer.zero_grad(set_to_none=True)


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=hw.device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))


if ddp:
    dist.destroy_process_group()
