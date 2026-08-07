"""Tiny in-process torch modules that mimic real HF naming — NO HF download.

Used by the torch-tagged test groups (discovery, activations, decile, pipeline).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Cfg:
    def __init__(self, model_type):
        self.model_type = model_type
        self.architectures = [model_type]


def tiny_llama(n_layers=2, d=32):
    """model.layers.{i}.self_attn.{q,k,v,o}_proj + mlp.{gate,up,down}_proj."""
    m = nn.Module()
    m.config = _Cfg("llama")
    m.model = nn.Module()
    m.model.embed_tokens = nn.Embedding(50, d)
    layers = nn.ModuleList()
    for _ in range(n_layers):
        blk = nn.Module()
        attn = nn.Module()
        attn.q_proj = nn.Linear(d, d, bias=False)
        attn.k_proj = nn.Linear(d, d, bias=False)
        attn.v_proj = nn.Linear(d, d, bias=False)
        attn.o_proj = nn.Linear(d, d, bias=False)
        blk.self_attn = attn
        mlp = nn.Module()
        mlp.gate_proj = nn.Linear(d, 4 * d, bias=False)
        mlp.up_proj = nn.Linear(d, 4 * d, bias=False)
        mlp.down_proj = nn.Linear(4 * d, d, bias=False)
        blk.mlp = mlp
        layers.append(blk)
    m.model.layers = layers
    m.lm_head = nn.Linear(d, 50, bias=False)
    return m


def tiny_pythia(n_layers=2, d=48):
    """gpt_neox.layers.{i}.attention.query_key_value (rows=3d), attention.dense,
    mlp.dense_h_to_4h, mlp.dense_4h_to_h."""
    m = nn.Module()
    m.config = _Cfg("gpt_neox")
    m.gpt_neox = nn.Module()
    m.gpt_neox.embed_in = nn.Embedding(50, d)
    layers = nn.ModuleList()
    for _ in range(n_layers):
        blk = nn.Module()
        attn = nn.Module()
        attn.query_key_value = nn.Linear(d, 3 * d, bias=True)   # fused QKV (rows=3d)
        attn.dense = nn.Linear(d, d, bias=True)
        blk.attention = attn
        mlp = nn.Module()
        mlp.dense_h_to_4h = nn.Linear(d, 4 * d, bias=True)
        mlp.dense_4h_to_h = nn.Linear(4 * d, d, bias=True)
        blk.mlp = mlp
        layers.append(blk)
    m.gpt_neox.layers = layers
    m.embed_out = nn.Linear(d, 50, bias=False)
    return m


class TinyCausalLM(nn.Module):
    """Runnable llama-named causal LM: embedding -> linear blocks -> tied head.

    forward(input_ids, labels=None) -> object with .logits and optional .loss.
    """
    def __init__(self, n_layers=2, d=32, vocab=50):
        super().__init__()
        self.config = _Cfg("llama")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        layers = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.Module()
            attn = nn.Module()
            attn.q_proj = nn.Linear(d, d, bias=False)
            attn.k_proj = nn.Linear(d, d, bias=False)
            attn.v_proj = nn.Linear(d, d, bias=False)
            attn.o_proj = nn.Linear(d, d, bias=False)
            blk.self_attn = attn
            mlp = nn.Module()
            mlp.gate_proj = nn.Linear(d, 4 * d, bias=False)
            mlp.up_proj = nn.Linear(d, 4 * d, bias=False)
            mlp.down_proj = nn.Linear(4 * d, d, bias=False)
            blk.mlp = mlp
            layers.append(blk)
        self.model.layers = layers
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.d = d
        self.vocab = vocab

    def forward(self, input_ids, labels=None, **kw):
        h = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            a = blk.self_attn
            h = h + a.o_proj(a.q_proj(h) + a.k_proj(h) + a.v_proj(h))
            mlp = blk.mlp
            h = h + mlp.down_proj(F.relu(mlp.gate_proj(h)) * mlp.up_proj(h))
        logits = self.lm_head(h)
        out = type("Out", (), {})()
        out.logits = logits
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out.loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab), shift_labels.view(-1))
        return out


class TinyPythiaCausalLM(nn.Module):
    """Runnable pythia-named causal LM with fused QKV (model-swap smoke test)."""
    def __init__(self, n_layers=2, d=48, vocab=50):
        super().__init__()
        self.config = _Cfg("gpt_neox")
        self.gpt_neox = nn.Module()
        self.gpt_neox.embed_in = nn.Embedding(vocab, d)
        layers = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.Module()
            attn = nn.Module()
            attn.query_key_value = nn.Linear(d, 3 * d, bias=True)
            attn.dense = nn.Linear(d, d, bias=True)
            blk.attention = attn
            mlp = nn.Module()
            mlp.dense_h_to_4h = nn.Linear(d, 4 * d, bias=True)
            mlp.dense_4h_to_h = nn.Linear(4 * d, d, bias=True)
            blk.mlp = mlp
            layers.append(blk)
        self.gpt_neox.layers = layers
        self.embed_out = nn.Linear(d, vocab, bias=False)
        self.d = d
        self.vocab = vocab

    def forward(self, input_ids, labels=None, **kw):
        h = self.gpt_neox.embed_in(input_ids)
        for blk in self.gpt_neox.layers:
            qkv = blk.attention.query_key_value(h)
            q, k, v = qkv.chunk(3, dim=-1)
            h = h + blk.attention.dense(q + k + v)
            h = h + blk.mlp.dense_4h_to_h(F.relu(blk.mlp.dense_h_to_4h(h)))
        logits = self.embed_out(h)
        out = type("Out", (), {})()
        out.logits = logits
        if labels is not None:
            sl = logits[:, :-1, :].contiguous()
            slb = labels[:, 1:].contiguous()
            out.loss = F.cross_entropy(sl.view(-1, self.vocab), slb.view(-1))
        return out


def tiny_causal_lm(n_layers=2, d=32, vocab=50):
    return TinyCausalLM(n_layers, d, vocab)
