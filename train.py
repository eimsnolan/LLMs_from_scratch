from logging import config
import torch

from bigram import BigramConfig, BigramConfigTest, BigramLanguageModel
from gpt import GPTConfig, GPTConfigTest, GPT


# Altered version of : https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s
# at timestamp 1hr 24mins
# Decoder only transformer, no cross attention or encoder

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a map from chars to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for  i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# forst 90% wil be train data
data = torch.tensor(encode(text), dtype = torch.long)
n= int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate small batch of inputs x and outputs y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, ( config.batch_size,))
    x = torch.stack([data[i:i+ config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+ config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros( config.eval_iters)
        for k in range( config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


print("basing vocab size of Shakespeare dataset")
config_args = {}
config_args['vocab_size'] = vocab_size

#config = BigramConfigTest(**config_args)
#model = BigramLanguageModel(config)


config = GPTConfigTest(**config_args)
model = GPT(config)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)

# training loop
for iter in range(config.max_iters):

    # every onece in a while evaluate loss on train and val sets 
    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']}, val loss {losses['val']}")

    # sample a batch of data
    xb, yb, = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens = 300)[0].tolist()))