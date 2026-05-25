"""
================================================================================
nanoGPT model.py — ANNOTATED WALKTHROUGH
================================================================================

This is a copy of Karpathy's model.py with extensive inline explanations.
The actual code is unchanged. Comments marked with `# >>>` are the annotations.
Run `train.py` against the *original* model.py — this file is for reading.

--------------------------------------------------------------------------------
BIG PICTURE — what is a GPT, in one paragraph
--------------------------------------------------------------------------------
A GPT (Generative Pre-trained Transformer) is a function that takes a sequence
of token IDs and outputs, for each position, a probability distribution over
"what token comes next". You train it by feeding billions of text snippets and
asking it to predict each next token. The architecture is just:

    tokens -> embeddings -> [Block] x N -> LayerNorm -> Linear -> logits

Each Block is two sublayers wrapped in residual connections:
    1. Causal self-attention (lets each position look at earlier positions)
    2. A feedforward MLP (does per-position computation)

That's it. The model has no recurrence, no convolutions, no memory other than
the attention window. Everything interesting comes from stacking many Blocks
and scaling everything up. GPT-2 small = 12 Blocks, 768-dim embeddings, 124M
params. GPT-3 = 96 Blocks, 12288-dim, 175B params. Same architecture, more of it.

--------------------------------------------------------------------------------
DATA FLOW for one training step (batch_size=B, seq_len=T, embed_dim=C)
--------------------------------------------------------------------------------
  Input:  idx of shape (B, T) — token IDs, integers in [0, vocab_size)
          targets of shape (B, T) — the *next* token at each position

  Step 1: token + position embeddings -> (B, T, C)
  Step 2: stack of N transformer Blocks, each (B, T, C) -> (B, T, C)
  Step 3: final LayerNorm -> (B, T, C)
  Step 4: lm_head Linear -> (B, T, vocab_size)  [logits]
  Step 5: cross_entropy(logits, targets) -> scalar loss
  Step 6: loss.backward(), optimizer.step()
================================================================================
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ==============================================================================
# LayerNorm
# ==============================================================================
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    # >>> LayerNorm normalizes a vector to have mean=0 and variance=1, then
    # >>> applies a learnable scale (weight) and shift (bias). It is applied
    # >>> per-token (independently across the C feature dimensions of each
    # >>> token's vector). It stabilizes training by keeping activations from
    # >>> drifting in magnitude as they pass through many layers.
    # >>>
    # >>> WHY a custom class instead of nn.LayerNorm? Only to allow bias=False.
    # >>> Removing the bias gives a tiny speedup and slightly better results
    # >>> per the LLaMA paper. PyTorch's built-in nn.LayerNorm forces a bias.
    # >>>
    # >>> Note: modern LLMs (LLaMA, GPT-NeoX) use RMSNorm — same idea but skips
    # >>> the mean-subtraction step. Slightly cheaper, comparable quality.
    # >>> nanoGPT stays with classic LayerNorm to match GPT-2.

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # learnable scale,  init=1
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # learnable shift, init=0

    def forward(self, input):
        # >>> 1e-5 is the epsilon added to variance for numerical stability
        # >>> (so we never divide by zero on a constant input).
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ==============================================================================
# CausalSelfAttention — the core mechanism that makes transformers work
# ==============================================================================
class CausalSelfAttention(nn.Module):
    # >>> SELF-ATTENTION: each token decides which other tokens to look at,
    # >>> then takes a weighted average of their values. The weights depend
    # >>> on a similarity score between a "query" vector (from this token)
    # >>> and a "key" vector (from each other token). The result is a
    # >>> context-aware representation of this token.
    # >>>
    # >>> CAUSAL: a token at position t can only attend to positions <= t.
    # >>> This is the magic that makes GPT able to predict next tokens:
    # >>> at training time, we feed in the whole sequence at once but mask
    # >>> the attention so position t sees only tokens 0..t, exactly as it
    # >>> would at inference time when generating one token at a time.
    # >>>
    # >>> MULTI-HEAD: instead of one big attention, we split the embedding
    # >>> dim C across n_head separate attention computations and concatenate
    # >>> the results. Each head can specialize on different patterns
    # >>> (syntax, long-range deps, specific word relationships, etc.).

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # heads must divide evenly into embed dim

        # >>> One Linear layer produces Q, K, and V for all heads at once.
        # >>> Output is 3*C wide; we split it into three C-wide tensors later.
        # >>> Stacking Q,K,V into one matmul is faster than three separate ones.
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # >>> After attention, this projects the C-wide result back to C
        # >>> (i.e. "mixes" the heads). Without it, heads would be independent.
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # >>> Dropout for regularization. During training, randomly zeroes a
        # >>> fraction of values to prevent overfitting. nanoGPT defaults to 0
        # >>> (no dropout) for small models because they underfit anyway.
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # >>> "Flash Attention" is a fused CUDA kernel (Tri Dao, 2022) that
        # >>> computes attention without materializing the full T×T attention
        # >>> matrix in memory. For block_size=1024 and n_head=12, the naive
        # >>> version stores a 12×1024×1024 tensor per batch element; Flash
        # >>> Attention computes the same result tile-by-tile and is ~3x faster
        # >>> and uses far less memory. PyTorch ships it as F.scaled_dot_product_attention.
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # >>> The fallback path needs an explicit causal mask: a lower-
            # >>> triangular matrix of 1s. tril() = "triangular lower". We
            # >>> register it as a buffer so it moves to GPU with .to(device)
            # >>> but is not a learnable parameter.
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # >>> x is (B, T, C). At each of T positions in each of B sequences,
        # >>> we have a C-dimensional vector. We want to produce a new (B, T, C).
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # >>> ONE matmul produces all of Q, K, V across all heads:
        # >>> c_attn(x) is (B, T, 3*C); .split(C, dim=2) gives three (B, T, C).
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # >>> Reshape (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size).
        # >>> head_size = C // n_head. We move n_head to be next to B so that
        # >>> the subsequent batched matmul does attention independently per head.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # >>> Flash Attention does everything: scaled dot product, causal
            # >>> masking (is_causal=True), softmax, dropout, and the matmul
            # >>> with V — all in one fused kernel. Returns (B, nh, T, hs).
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # >>> THE NAIVE / EDUCATIONAL PATH — what's actually happening:
            # >>>   1. q @ k.transpose: (B,nh,T,hs) x (B,nh,hs,T) -> (B,nh,T,T)
            # >>>      For each token i, this gives a similarity score with
            # >>>      every other token j. Dividing by sqrt(hs) is the
            # >>>      "scaled" in "scaled dot product attention" — keeps the
            # >>>      scores from getting huge when hs is large, which would
            # >>>      make softmax saturate and gradients vanish.
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # >>>   2. Apply the causal mask: set the upper triangle to -inf
            # >>>      so softmax assigns those positions zero probability.
            # >>>      Token i can attend only to tokens 0..i (left-to-right).
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # >>>   3. Softmax along the last dim turns scores into a probability
            # >>>      distribution over previous tokens (rows sum to 1).
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # >>>   4. Weighted sum of values: (B,nh,T,T) @ (B,nh,T,hs) -> (B,nh,T,hs).
            # >>>      For each token i, output[i] = sum over j of att[i,j] * v[j].
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # >>> Concatenate heads: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C).
        # >>> .contiguous() because transpose makes the tensor non-contiguous in
        # >>> memory and .view() needs contiguous memory.
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # >>> Final projection mixes information across heads. Without this,
        # >>> each head's output would live in its own slice of the embedding.
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# ==============================================================================
# MLP — the position-wise feedforward network
# ==============================================================================
class MLP(nn.Module):
    # >>> Attention mixes information ACROSS positions. The MLP processes
    # >>> each position INDEPENDENTLY. It's just: project up to 4*C, apply
    # >>> GELU nonlinearity, project back down to C.
    # >>>
    # >>> WHY 4x EXPANSION? It's the standard from "Attention Is All You Need"
    # >>> and has stuck for years. The wider intermediate gives the model more
    # >>> capacity to compute nonlinear transformations. Modern LLMs sometimes
    # >>> use ratios of 2.66x with SwiGLU activation (LLaMA) for similar
    # >>> quality at lower FLOPs. nanoGPT stays with 4x + GELU to match GPT-2.
    # >>>
    # >>> The MLP holds roughly 2/3 of the model's parameters. Attention is
    # >>> what makes transformers smart; MLP is where they store learned
    # >>> "knowledge" (per work on transformer interpretability).

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # expand
        self.gelu    = nn.GELU()                                                       # nonlinearity
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # contract
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # >>> Applied independently at every position. Shape stays (B, T, C).
        x = self.c_fc(x)
        x = self.gelu(x)  # GELU(x) ≈ x * Φ(x); smoother than ReLU, what GPT-2 used
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ==============================================================================
# Block — the unit you stack N times to build the transformer
# ==============================================================================
class Block(nn.Module):
    # >>> THE TRANSFORMER BLOCK. Each Block does:
    # >>>
    # >>>   x = x + attention(layernorm(x))   # mix across positions
    # >>>   x = x + mlp(layernorm(x))         # transform per-position
    # >>>
    # >>> The `x +` is a "residual connection": the block learns a DELTA on
    # >>> top of the input. This is critical — it lets gradients flow back
    # >>> through many layers without vanishing, and lets the model "skip"
    # >>> layers it doesn't need.
    # >>>
    # >>> PRE-LN vs POST-LN: nanoGPT uses Pre-LN (LayerNorm BEFORE each
    # >>> sublayer, residual unmodified). The original "Attention Is All You
    # >>> Need" paper used Post-LN. Pre-LN trains more stably without warmup
    # >>> and is now standard.

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # attention sublayer
        x = x + self.mlp(self.ln_2(x))   # MLP sublayer
        return x


# ==============================================================================
# GPTConfig — all the hyperparameters
# ==============================================================================
@dataclass
class GPTConfig:
    # >>> Default values here are GPT-2 small (124M params).
    block_size: int = 1024     # >>> max context length in tokens. Attention is O(T²).
    vocab_size: int = 50304    # >>> number of distinct tokens. GPT-2 BPE has 50257;
                               # >>> padding to a multiple of 64 makes matmuls faster on GPUs.
    n_layer: int = 12          # >>> number of stacked Blocks. The "depth" of the model.
    n_head: int = 12           # >>> number of attention heads per Block.
    n_embd: int = 768          # >>> embedding dimension. Determines per-head size = 768/12 = 64.
    dropout: float = 0.0       # >>> 0 for pretraining big models; 0.1+ for finetuning small ones.
    bias: bool = True          # >>> True matches GPT-2; False is a tiny bit better+faster (LLaMA-style).

    # >>> SIZING INTUITION (parameter count ≈ 12 * n_layer * n_embd² for transformer body):
    # >>>   tiny-shakespeare default:  6 layers,  6 heads,  384 dim ->  ~10M params
    # >>>   GPT-2 small:              12 layers, 12 heads,  768 dim -> ~124M params
    # >>>   GPT-2 medium:             24 layers, 16 heads, 1024 dim -> ~350M params
    # >>>   GPT-2 XL:                 48 layers, 25 heads, 1600 dim -> ~1.5B params
    # >>>   LLaMA-2 7B:               32 layers, 32 heads, 4096 dim, vocab=32k -> ~7B params


# ==============================================================================
# GPT — the full model
# ==============================================================================
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # >>> ModuleDict is just a dict whose values are nn.Modules; PyTorch
        # >>> tracks them for .parameters() and .to(device).
        self.transformer = nn.ModuleDict(dict(
            # >>> wte = Word Token Embedding. A lookup table: token_id -> C-dim vector.
            # >>> Shape (vocab_size, n_embd). Each row is the "meaning" of one token,
            # >>> learned from scratch during training.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # >>> wpe = Word Position Embedding. A separate learned vector per
            # >>> position 0..block_size-1. Added to the token embedding so the
            # >>> model knows token order (attention itself is permutation-invariant).
            # >>> Modern alternatives: sinusoidal (original paper), RoPE (LLaMA),
            # >>> ALiBi. nanoGPT keeps it simple with learned absolute positions.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # >>> h = the stack of Blocks. ModuleList = ordered list of modules.
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # >>> Final LayerNorm before the lm_head. Important: stabilizes the
            # >>> last activations before projecting to vocab_size logits.
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # >>> lm_head = Language Model Head. Linear projection from C to vocab_size
        # >>> producing one logit per possible next token at each position.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # >>> WEIGHT TYING: the input embedding (wte) and output projection
        # >>> (lm_head) share the same weight matrix. This saves ~38M params
        # >>> in GPT-2 small (50257 * 768) and slightly improves quality. The
        # >>> intuition: "the vector for token X" should be the same whether
        # >>> you're embedding token X going in or scoring token X coming out.
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # >>> Initialize all Linear and Embedding weights from N(0, 0.02).
        # >>> This is the GPT-2 init scheme.
        # init all weights
        self.apply(self._init_weights)
        # >>> SCALED INIT for residual projections: the projection at the END
        # >>> of each sublayer (c_proj in attention and MLP) is scaled down by
        # >>> 1/sqrt(2 * n_layer). This keeps the variance of activations from
        # >>> growing as they pass through many residual additions. Per the
        # >>> GPT-2 paper, this matters for deep models.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # >>> Convention from the GPT-2 paper: parameter counts usually exclude
        # >>> embedding params because they don't do "computation", they're
        # >>> just a lookup. Token embeddings are double-duty (also lm_head)
        # >>> so they're kept; position embeddings are subtracted.
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # >>> Called by self.apply() above on every submodule.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --------------------------------------------------------------------------
    # forward — THE TRAINING STEP
    # --------------------------------------------------------------------------
    def forward(self, idx, targets=None):
        # >>> idx:     (B, T) integer token IDs
        # >>> targets: (B, T) integer token IDs — the NEXT token at each position.
        # >>>          If None, we're in inference mode and skip loss computation.
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # >>> Position indices [0, 1, 2, ..., t-1]. Same for every batch element.
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        # >>> Look up the C-dim vector for each token.   Shape: (B, T, C)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # >>> Look up the C-dim vector for each position. Shape: (T, C)
        # >>> Broadcasts across batch when added to tok_emb.
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # >>> ADD them (don't concatenate). This is the key design choice that
        # >>> lets us inject position info without growing the vector size.
        x = self.transformer.drop(tok_emb + pos_emb)

        # >>> Pass through every Block. Shape stays (B, T, C) throughout.
        for block in self.transformer.h:
            x = block(x)

        # >>> Final LayerNorm before producing logits.
        x = self.transformer.ln_f(x)

        if targets is not None:
            # >>> TRAINING PATH: produce logits at EVERY position, compute loss.
            # >>> logits shape: (B, T, vocab_size)
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # >>> Flatten (B, T, V) -> (B*T, V) and (B, T) -> (B*T) for cross_entropy.
            # >>> ignore_index=-1 lets you mark padding positions with -1 to exclude them.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # >>> INFERENCE PATH (called from generate()): we only need logits
            # >>> at the LAST position to sample the next token. Skip computing
            # >>> the other T-1 sets of logits — big speedup at long contexts.
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # >>> Model surgery: shrink the context window after construction.
        # >>> Useful when loading pretrained weights but wanting a smaller
        # >>> context for memory reasons.
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # >>> Loads GPT-2 weights from HuggingFace and copies them into this
        # >>> architecture. Plumbing — useful for finetuning on top of GPT-2.
        # >>> Skim this on first read; the interesting stuff is above.
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # >>> Sets up AdamW optimizer with two parameter groups:
        # >>>   - 2D tensors (matmul weights, embeddings): weight decay applied
        # >>>   - 1D tensors (biases, LayerNorm gains): NO weight decay
        # >>> This split is standard practice — decaying biases/LN gains hurts.
        # >>> "fused" AdamW is a CUDA-optimized version, faster than default.
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # >>> MFU = "Model FLOPs Utilization". What fraction of the GPU's
        # >>> theoretical peak FLOPs are we actually using? A well-tuned
        # >>> training run hits 30-50% MFU. Less means the GPU is starved
        # >>> (data loading, kernel launch overhead, suboptimal batch size).
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # --------------------------------------------------------------------------
    # generate — autoregressive text generation
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # >>> This is "autoregressive sampling": predict one token, append it
        # >>> to the context, predict the next one, repeat. Naive (O(T²) per
        # >>> step). Production systems use a KV cache to make this O(T).
        # >>>
        # >>> TEMPERATURE: scales logits before softmax.
        # >>>   T < 1.0: sharpens distribution -> more deterministic / "focused"
        # >>>   T = 1.0: sample from the true model distribution
        # >>>   T > 1.0: flattens distribution -> more random / "creative"
        # >>>
        # >>> TOP-K: before sampling, keep only the K most likely tokens and
        # >>> set the rest to -inf. Prevents the model from picking absurdly
        # >>> low-probability tokens that, summed, can still get sampled.
        for _ in range(max_new_tokens):
            # >>> If the context exceeds block_size, crop to the most recent
            # >>> block_size tokens. The model literally cannot see further.
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # >>> Forward pass. Returns logits of shape (B, 1, vocab_size)
            # >>> because targets=None triggers the last-position-only optimization.
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # >>> Squeeze the time dim and scale by temperature.
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # >>> multinomial = sample 1 token from the probability distribution.
            # >>> Use torch.argmax(probs) instead for fully greedy decoding.
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # >>> Append and loop.
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ==============================================================================
# RECAP — what you just read
# ==============================================================================
# Building blocks, smallest to largest:
#
#   LayerNorm           ~20 lines    normalize a vector, learnable scale/shift
#   CausalSelfAttention ~50 lines    Q/K/V projection + masked softmax + mix heads
#   MLP                 ~15 lines    Linear -> GELU -> Linear, with 4x expansion
#   Block               ~10 lines    LN -> attn -> residual; LN -> mlp -> residual
#   GPT                 ~50 lines    embeddings + N Blocks + final LN + lm_head
#
# The actual GPT-2 architecture is ~150 lines of substantive code. Everything
# else in this file (~300 more lines) is plumbing: loading weights, optimizers,
# inference helpers, parameter counting.
#
# Scaling this up is mostly a matter of:
#   - More layers (n_layer)
#   - Wider embeddings (n_embd)
#   - More heads (n_head)
#   - Larger context (block_size) — costs O(T²) memory
#   - Better data
#
# The architecture has barely changed since 2018. What changed: data quantity,
# data quality, compute, and small tweaks (Pre-LN, RMSNorm, RoPE, SwiGLU,
# Grouped-Query Attention). The 1500-line modeling files in HuggingFace are
# mostly handling the long tail of: multiple checkpoint formats, distributed
# training, KV caching, quantization, etc. The CORE is what you see here.
#
# WHAT TO READ NEXT to deepen understanding:
#   - train.py        — the training loop: AdamW, gradient clipping, LR schedule
#   - sample.py       — calls generate() with various prompts
#   - "The Illustrated Transformer" by Jay Alammar (Google for it)
#   - Andrej Karpathy's "Let's build GPT" YouTube video (2 hours, builds this
#     entire file live from scratch)
# ==============================================================================
