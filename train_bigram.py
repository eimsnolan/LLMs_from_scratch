from re import L
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import functional as F

# Altered version of : https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s
# at timestamp 1hr 24mins

# hyperparameters
batch_size = 32 # sequences in parralel
block_size = 8 # context_length
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3 # attention can't deal with a high learning rate wel 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# self attention 
class Head(nn.Module):
    """one head of self attention

    Args:
        nn (_type_): _description_
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril isnt a parameter of the model so you have to do this
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B, T, C = x.shape # C is attention head size!! 

        # let's see a single Head perform self-attention
        #head_size = 16
        # all the queries dot product with all the keys
        # wei = affinities between keys and queries
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # the scaling vector of 1/sqr(head size) is too make sure softmax doesn't turn out to be sharp 
        # from the original attention paper 
        wei =  q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)

        tril = torch.tril(torch.ones(T, T))
        # this below line doesn't allow all the nodes to talk to each other (auto regressive for text generation)
        # (nodes from the future can't talk to nodes from the past)
        # if you want all the nodes to talk to one another e.g. sentiment analysis
        # then delete this masked_fill line 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # weighted agregation
        v = self.value(x) # (B, T, C) vectors we aggregate, not the raw x tokens 
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# still not great results - add mulitple heads of attention!
class MultiHeadAttention(nn.Module):
    """ adding mulitple heads of attention in parallel
    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


# bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8 dimensional self attention
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,t) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C = embed C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.sa_heads(x) # apply one head of self attention (B, T ,C)
        # add linear layer
        logits = self.lm_head(x) # (B, T, vocab size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to make it not more than block size else our positional embeddings
            # table will run out of space 
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # every onece in a while evaluate loss on train and val sets 
    if iter % eval_interval == 0:
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