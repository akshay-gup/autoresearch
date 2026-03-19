"""
Autoresearch pretraining script. Single-file training that automatically uses all visible NVIDIA GPUs.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import get_kernel

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


@dataclass
class TrainRecipe:
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"
    total_batch_size: int = 2**19
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple[float, float] = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    depth: int = 8
    device_batch_size: int = 128

    @classmethod
    def from_json(cls, path: str | None):
        if not path:
            return cls()
        data = json.loads(Path(path).read_text())
        if "adam_betas" in data:
            data["adam_betas"] = tuple(data["adam_betas"])
        return cls(**data)

    def to_json_dict(self):
        data = asdict(self)
        data["adam_betas"] = list(self.adam_betas)
        return data

    def recipe_hash(self):
        payload = json.dumps(self.to_json_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos.bfloat16()[None, :, None, :], sin.bfloat16()[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel() + value_embeds_numel + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        logits = self.lm_head(norm(x)).float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
        return logits


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / bias1))


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(p, grad, state["exp_avg"], state["exp_avg_sq"], self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t, self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params, shape, device, dtype = len(params), p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params, state["momentum_buffer"], state["second_momentum_buffer"], self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


def build_model_config(depth, recipe, vocab_size):
    base_dim = depth * recipe.aspect_ratio
    model_dim = ((base_dim + recipe.head_dim - 1) // recipe.head_dim) * recipe.head_dim
    num_heads = model_dim // recipe.head_dim
    return GPTConfig(sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size, n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim, window_pattern=recipe.window_pattern)


def get_lr_multiplier(progress, recipe):
    if progress < recipe.warmup_ratio:
        return progress / recipe.warmup_ratio if recipe.warmup_ratio > 0 else 1.0
    if progress < 1.0 - recipe.warmdown_ratio:
        return 1.0
    cooldown = (1.0 - progress) / recipe.warmdown_ratio
    return cooldown + (1 - cooldown) * recipe.final_lr_frac


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress, recipe):
    return recipe.weight_decay * (1 - progress)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_checkpoint_if_present(model, optimizer, path, device):
    if not path:
        return {"step": 0, "total_training_time": 0.0}
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("trainer", {"step": 0, "total_training_time": 0.0})


def save_checkpoint(path, model, optimizer, trainer_state, recipe, config):
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "trainer": trainer_state,
        "recipe": recipe.to_json_dict(),
        "config": asdict(config),
    }, output_path)


def run_training(args):
    t_start = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("Training requires at least one NVIDIA GPU")
    device = torch.device("cuda:0")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    H100_BF16_PEAK_FLOPS = 989.5e12
    recipe = TrainRecipe.from_json(args.recipe)
    if args.export_recipe:
        Path(args.export_recipe).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export_recipe).write_text(json.dumps(recipe.to_json_dict(), indent=2) + "\n")
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")
    config = build_model_config(recipe.depth, recipe, vocab_size)
    print(f"Model config: {asdict(config)}")
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    base_model = unwrap_model(model)
    param_counts = base_model.num_scaling_params()
    num_params = param_counts["total"]
    num_flops_per_token = base_model.estimate_flops()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    tokens_per_fwdbwd = recipe.device_batch_size * MAX_SEQ_LEN * num_gpus
    assert recipe.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = recipe.total_batch_size // tokens_per_fwdbwd
    optimizer = base_model.setup_optimizer(
        unembedding_lr=recipe.unembedding_lr,
        embedding_lr=recipe.embedding_lr,
        scalar_lr=recipe.scalar_lr,
        adam_betas=recipe.adam_betas,
        matrix_lr=recipe.matrix_lr,
        weight_decay=recipe.weight_decay,
    )
    trainer_state = load_checkpoint_if_present(model, optimizer, args.load_checkpoint, device)
    compiled_model = model if num_gpus > 1 else torch.compile(model, dynamic=False)
    global_batch_size = recipe.device_batch_size * num_gpus
    train_loader = make_dataloader(tokenizer, global_batch_size, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)
    time_budget = args.time_budget or TIME_BUDGET
    print(f"Time budget: {time_budget}s")
    print(f"Using {num_gpus} GPU(s) with per-GPU batch size {recipe.device_batch_size} (global batch size {global_batch_size})")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = float(trainer_state.get("total_training_time", 0.0))
    step = int(trainer_state.get("step", 0))
    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        for _micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = compiled_model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)
        progress = min(total_training_time / time_budget, 1.0)
        lrm = get_lr_multiplier(progress, recipe)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress, recipe)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        compiled_model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        if math.isnan(train_loss_f) or train_loss_f > 100:
            raise RuntimeError("Training failed: NaN or exploding loss")
        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 10:
            total_training_time += dt
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(recipe.total_batch_size / dt)
        mfu = 100 * num_flops_per_token * recipe.total_batch_size / dt / H100_BF16_PEAK_FLOPS
        remaining = max(0, time_budget - total_training_time)
        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)
        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()
        step += 1
        if step > 10 and total_training_time >= time_budget:
            break
    print()
    total_tokens = step * recipe.total_batch_size
    compiled_model.eval()
    with autocast_ctx:
        selection_bpb = evaluate_bpb(compiled_model, tokenizer, global_batch_size, split=args.selection_split)
        report_bpb = evaluate_bpb(compiled_model, tokenizer, global_batch_size, split=args.report_split)
    t_end = time.time()
    startup_time = t_start_training - t_start
    steady_state_mfu = 100 * num_flops_per_token * recipe.total_batch_size * max(step - 10, 0) / max(total_training_time, 1e-6) / H100_BF16_PEAK_FLOPS
    peak_vram_mb = max(torch.cuda.max_memory_allocated(i) for i in range(num_gpus)) / 1024 / 1024
    trainer_state = {"step": step, "total_training_time": total_training_time, "epoch": epoch}
    save_checkpoint(args.save_checkpoint, model, optimizer if args.save_optimizer else None, trainer_state, recipe, config)
    metrics = {
        "selection_bpb": selection_bpb,
        "report_bpb": report_bpb,
        "val_bpb": selection_bpb,
        "training_seconds": total_training_time,
        "total_seconds": t_end - t_start,
        "startup_seconds": startup_time,
        "peak_vram_mb": peak_vram_mb,
        "mfu_percent": steady_state_mfu,
        "total_tokens": total_tokens,
        "num_steps": step,
        "num_params": num_params,
        "depth": recipe.depth,
        "recipe_hash": recipe.recipe_hash(),
        "selection_split": args.selection_split,
        "report_split": args.report_split,
    }
    print("---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    if args.metrics_json:
        output_path = Path(args.metrics_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an autoresearch training experiment")
    parser.add_argument("--recipe", type=str, default=None, help="Path to a recipe JSON file")
    parser.add_argument("--export-recipe", type=str, default=None, help="Write the effective recipe JSON to this path")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Load model/optimizer state from a checkpoint")
    parser.add_argument("--save-checkpoint", type=str, default=None, help="Write checkpoint to this path at the end")
    parser.add_argument("--save-optimizer", action="store_true", help="Persist optimizer state inside the checkpoint")
    parser.add_argument("--metrics-json", type=str, default=None, help="Write final metrics JSON to this path")
    parser.add_argument("--time-budget", type=int, default=None, help="Override training time budget in seconds")
    parser.add_argument("--selection-split", type=str, default="selection", choices=["train", "selection", "report", "val"], help="Split used for evolutionary selection")
    parser.add_argument("--report-split", type=str, default="report", choices=["train", "selection", "report", "val"], help="Untouched holdout split for reporting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_training(parser.parse_args())
