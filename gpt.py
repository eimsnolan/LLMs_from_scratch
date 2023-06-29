from dataclasses import dataclass
import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Altered version of : https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s
# at timestamp 1hr 24mins
# Decoder only transformer, no cross attention or encoder

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# self attention 
class Head(nn.Module):
    """one head of self attention
    """
    def __init__(self,  config, head_size):
        super().__init__()
        self.key = nn.Linear( config.n_embd,  head_size, bias=config.bias)
        self.query = nn.Linear( config.n_embd,  head_size, bias=config.bias)
        self.value = nn.Linear( config.n_embd,  head_size, bias=config.bias)
        # tril isnt a parameter of the model so you have to do this
        self.register_buffer('tril', torch.tril(torch.ones( config.block_size,  config.block_size)))
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        B, T, C = x.shape # C is attention head size!! 

        # let's see a single Head perform self-attention
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

    """
    def __init__(self,  config):
        super().__init__()
        head_size =  config.n_embd// config.n_head
        assert config.n_embd % config.n_head == 0
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range( config.n_head)])
        self.proj = nn.Linear( config.n_embd,  config.n_embd, bias=config.bias) # part of the skip connections, inof to be projected back in 
        self.dropout = nn.Dropout( config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # part of the skip connections, inof to be projected back in 
        return out


class MLP(nn.Module):
    def __init__(self,  config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( config.n_embd, 4 *  config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 *  config.n_embd,  config.n_embd, bias=config.bias), # part of the skip connections, info to be projected back in 
            nn.Dropout( config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transformer block, communication followed by computation """
    def __init__(self,  config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.sa = MultiHeadAttention( config) # i.e. 4 heads of 8 dimensional self attention
        self.ln2 = LayerNorm( config.n_embd, bias=config.bias) # normalises column 
        self.mlp = MLP(config)


    def forward(self, x):
        # we're adding x to itself as a skip connection: time 1hr 30 mins
        x = x + self.sa(self.ln1(x)) # apply one head of self attention (B, T ,C)
        x = x + self.mlp(self.ln2(x)) # feed forward MLp
        return x


# gpt language model
class GPT(nn.Module):

    def __init__(self,  config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding = nn.Embedding( config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding( config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range( config.n_layer)])
        self.ln_f = LayerNorm( config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear( config.n_embd,  config.vocab_size, bias=False)

        # initialize weights 
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        params = self.measure_params()
        print(f"number of parameters: {params/1e6} M")


    def measure_params(self, non_embeddings = True):
        """Calculate number of parameters of the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embeddings:
            n_params -= self.position_embedding.weight.numel()
        return n_params

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
        tok_emb = self.token_embedding(idx) # (B, T, C = embed C)
        pos_emb = self.position_embedding(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb
        # adding a block of attention + computation
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # improve optimizer with warm up scheduler
    def optimizer(self, device_type: str= 'cpu'):
        # refresher here: https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
        # get all params that require grad
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # get decay parameters
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        dict_params_decay = {p for n, p in params.items() if p.dim()>=2}
        dict_params_non_decay = {p for n, p in params.items() if p.dim()<2}

        # optim group for adam optimizer
        optim_group =list( [
            {'params': dict_params_decay, "weight_decay": self.config.weight_decay},
            {'params': dict_params_non_decay, "weight_decay": 0.0}
        ])

        num_decay_params = sum(p.numel() for p in dict_params_decay)
        num_nodecay_params = sum(p.numel() for p in dict_params_non_decay)
        print(f"num decayed parameter tensors: {len(dict_params_decay)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(dict_params_non_decay)}, with {num_nodecay_params:,} parameters")

        # check if we can used fused adam (need torch 2.0 and cuda)
        isFused = device_type == 'cuda'

        optimizer = torch.optim.AdamW(dict_params_decay,
                                       lr=self.config.learning_rate,
                                         betas = [self.config.beta1, self.config.beta2],
                                           weight_decay = self.config.weight_decay,
                                            fused = isFused)

        return optimizer

    @classmethod
    def load_pretrained(cls, model_type, override_args):
        assert model_type in {'gpt2', "gpt2-medium", 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


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
class GPTConfig:
    # hyperparameters
    batch_size: int = 64 # sequences in parralel
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200

    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # for linear and layer norms

    # optimizer params
    learning_rate: float = 3e-4 # attention can't deal with a high learning rate wel 
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

    # learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 2000 # how many steps to warm up for
    lr_decay_iters: int = 600000 # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


# dataclass of smaller config arguments for testing on a cpu 
@dataclass
class GPTConfigTest:
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
    bias: bool = False
    # optimizer params
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
