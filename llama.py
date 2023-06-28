from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# self attention 
class Head(nn.Module):
    """one head of self attention
    """
    def __init__(self,  config, head_size):
        super().__init__()
        self.key = nn.Linear( config.n_embd,  head_size, bias=False)
        self.query = nn.Linear( config.n_embd,  head_size, bias=False)
        self.value = nn.Linear( config.n_embd,  head_size, bias=False)
        # tril isnt a parameter of the model so you have to do this
        self.register_buffer('tril', torch.tril(torch.ones( config.block_size,  config.block_size)))
        self.dropout = nn.Dropout( config.dropout)


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
        wei = self.dropout(wei)
        # weighted aggregation
        v = self.value(x) # (B, T, C) vectors we aggregate, not the raw x tokens 
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# still not great results - add mulitple heads of attention!
class MultiHeadAttention(nn.Module):
    """ adding mulitple heads of attention in parallel
    Args:
        nn (_type_): _description_
    """
    def __init__(self,  config):
        super().__init__()
        head_size =  config.n_embd// config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range( config.n_head)])
        self.proj = nn.Linear( config.n_embd,  config.n_embd) # part of the skip connections, inof to be projected back in 
        self.dropout = nn.Dropout( config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # part of the skip connections, inof to be projected back in 
        return out


class MLP(nn.Module):
    def __init__(self,  config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( config.n_embd, 4 *  config.n_embd),
            nn.GELU(),
            nn.Linear(4 *  config.n_embd,  config.n_embd), # part of the skip connections, info to be projected back in 
            nn.Dropout( config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transformer block, communication followed by computation """
    def __init__(self,  config):
        super().__init__()
        self.ln1 = nn.LayerNorm( config.n_embd) # normalises column see LayerNorm1d for eg
        self.sa = MultiHeadAttention( config) # i.e. 4 heads of 8 dimensional self attention
        self.ln2 = nn.LayerNorm( config.n_embd) # normalises column 
        self.mlp = MLP(config)


    def forward(self, x):
        # we're adding x to itself as a skip connection: time 1hr 30 mins
        x = x + self.sa(self.ln1(x)) # apply one head of self attention (B, T ,C)
        x = x + self.mlp(self.ln2(x)) # feed forward MLp
        return x


# gpt language model
class Llama(nn.Module):

    def __init__(self,  config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding( config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding( config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range( config.n_layer)])
        self.ln_f = nn.LayerNorm( config.n_embd)
        self.lm_head = nn.Linear( config.n_embd,  config.vocab_size)

        # initialize weights 
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.measure_params()/1e6))

    def measure_params(self):
        """Calculate number of parameters of the model"""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        """Initialize weights of layers """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0, std = 0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0, std = 0.2)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,t) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C = embed C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb
        # adding a block of attention + computation
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1, top_k = None):
        """ Generates a sequence, make sure model is in evaluation mode"""
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to make it not more than block size else our positional embeddings
            # table will run out of space 
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            # scale by temperature
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)) )
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# dataclass of config arguments 
@dataclass
class LlamaConfig:
    # hyperparameters
    batch_size: int = 64 # sequences in parralel
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4 # attention can't deal with a high learning rate wel 
    eval_iters: int = 200

    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


# dataclass of smaller config arguments for testing on a cpu 
@dataclass
class LlamaConfigTest:
  # hyperparameters
    batch_size: int = 16 # sequences in parralel
    block_size: int = 32 # context_length
    vocab_size:int = 0
    dropout: float = 0.2
    max_iters: int = 3000
    eval_interval: int = 100
    n_head: int = 4
    n_layer: int = 4
    learning_rate: float = 1e-3 # attention can't deal with a high learning rate wel 
    eval_iters: int = 200
    n_embd = 64