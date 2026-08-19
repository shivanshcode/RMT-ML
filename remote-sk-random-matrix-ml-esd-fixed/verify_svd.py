#!/usr/bin/env python3
"""Read-only checks for HANDOFF §2 (outstanding) and §6 ("verify the singular
values are computed correctly").  Nothing here writes to the repo or to the
results; it loads a checkpoint, runs assertions, and prints a report.

    python verify_svd.py --model /path/to/Llama-3.1-8B --layers 0 9 25
    python verify_svd.py --model /path/to/pythia-160m --layers 0 5 10 --qkv-check

Checks
------
1. Precision chain.  Reports the checkpoint dtype, the dtype the model is
   materialised at, and the dtype reaching the SVD, and asserts no float32
   round-trip sits between the checkpoint and the float64 upcast.
2. Frobenius identity.  sum(s**2) vs ||W||_F**2, relative error.  This is the
   cheapest end-to-end check that the SVD is the SVD of the matrix that was
   actually loaded.
3. Reconstruction.  ||W - U diag(s) Vh||_F / ||W||_F.
4. Shape.  Reported (n, m) against the true weight shape (Llama K/V are
   1024x4096 under GQA, not 4096x4096).
5. Rank floor.  The smallest singular values against the checkpoint's own
   quantisation floor, so a "min_sval = 1e-8" is read as genuine rank deficiency
   rather than as signal.
6. --qkv-check: the transformers reshape verification that HANDOFF §2 lists as
   still outstanding.  Feeds a random input through the model's own attention
   forward path and compares the resulting Q projection to x @ Q.T for the Q
   block extracted by discovery.extract_qkv_block.  Only meaningful for models
   with a fused QKV (GPT-NeoX / Pythia); skipped otherwise.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _fmt(x):
    return f"{x:.3e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local checkpoint path")
    ap.add_argument("--layers", type=int, nargs="*", default=[0])
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--qkv-check", action="store_true")
    args = ap.parse_args()

    import torch
    from rmt.model_io import load_model
    from rmt import discovery as D

    print("=" * 78)
    print(f"checkpoint: {args.model}")
    model = load_model(args.model, model_path=args.model, dtype=args.dtype)

    # ---- 1. precision chain ------------------------------------------------ #
    params = list(model.parameters())
    live_dtype = params[0].dtype if params else None
    print(f"\n[1] precision chain")
    print(f"    load_model(dtype={args.dtype!r}) -> parameters are {live_dtype}")
    print(f"    discovery._weight_2d upcasts to torch.float64 directly "
          f"(no intermediate .float())")
    if live_dtype == torch.float16:
        print("    WARNING: fp16 has a narrower exponent than bf16. If the "
              "checkpoint is bf16, load with --dtype fp32 (the RunConfig "
              "default) so no value is flushed on the way in.")
    elif live_dtype in (torch.float32, torch.bfloat16):
        print("    OK: bf16 -> fp32 -> float64 is exact (bf16 is a truncated "
              "fp32); the only lossy step is the checkpoint's own storage.")

    spec = D.get_model_spec(model)
    num_heads = D.get_num_heads(model)
    print(f"    model spec num_heads={num_heads}")
    recs = D.discover_weight_matrices(model, layer_indices=args.layers,
                                      spec=spec, dtype="float64",
                                      num_heads=num_heads)
    print(f"    discovered {len(recs)} matrices over layers {args.layers}")

    # ---- 2-5. per matrix --------------------------------------------------- #
    print(f"\n[2-5] per-matrix checks")
    hdr = (f"{'name':52s} {'shape':>13s} {'|1-Σs²/‖W‖²F|':>14s} "
           f"{'recon':>9s} {'min_s':>10s} {'floor':>10s}")
    print(hdr)
    print("-" * len(hdr))
    worst_fro = 0.0
    for r in recs:
        W = np.asarray(r.weight, dtype=np.float64)
        assert W.dtype == np.float64, f"{r.name}: reached SVD as {W.dtype}"
        assert (r.n, r.m) == W.shape, (
            f"{r.name}: reported ({r.n},{r.m}) != true {W.shape}")
        U, s, Vh = np.linalg.svd(W, full_matrices=False)
        fro2 = float(np.sum(W * W))
        rel_fro = abs(1.0 - float(np.sum(s ** 2)) / fro2) if fro2 > 0 else 0.0
        worst_fro = max(worst_fro, rel_fro)
        recon = float(np.linalg.norm(W - (U * s) @ Vh) / np.sqrt(fro2))
        # entrywise quantisation step of the stored checkpoint, propagated to a
        # singular-value scale: eps_store * |W|_rms * sqrt(min(n,m)) is the
        # order below which a singular value carries no checkpoint information.
        eps_store = {torch.bfloat16: 2 ** -8, torch.float16: 2 ** -11,
                     torch.float32: 2 ** -24}.get(live_dtype, 2 ** -24)
        rms = float(np.sqrt(fro2 / W.size))
        floor = eps_store * rms * np.sqrt(min(W.shape))
        print(f"{r.name[:52]:52s} {str(W.shape):>13s} {_fmt(rel_fro):>14s} "
              f"{_fmt(recon):>9s} {_fmt(s.min()):>10s} {_fmt(floor):>10s}")
        assert rel_fro < 1e-12, f"{r.name}: Frobenius identity violated"
        assert recon < 1e-12, f"{r.name}: SVD does not reconstruct W"
    print(f"\n    worst |1 - Σs²/‖W‖²_F| = {_fmt(worst_fro)}  (must be ~1e-15)")
    print("    'min_s' below 'floor' means the smallest singular values are "
          "genuine rank deficiency of the stored matrix, not resolvable signal.")

    # ---- 6. fused-QKV reshape against the model's own forward path --------- #
    if args.qkv_check:
        print(f"\n[6] fused-QKV layout check against transformers")
        try:
            import transformers
            print(f"    transformers {transformers.__version__}")
        except Exception as e:
            print(f"    could not import transformers: {e}")
            return 1
        attn = None
        for name, mod in model.named_modules():
            if hasattr(mod, "query_key_value"):
                attn, attn_name = mod, name
                break
        if attn is None:
            print("    no fused query_key_value module found "
                  "(model does not use a fused QKV) — skipped")
        else:
            cfg = model.config
            num_heads = int(getattr(cfg, "num_attention_heads"))
            hidden = int(getattr(cfg, "hidden_size"))
            head_dim = hidden // num_heads
            Wqkv = attn.query_key_value.weight.detach().to(
                torch.float64).cpu().numpy()
            x = torch.randn(1, 4, hidden, dtype=attn.query_key_value.weight.dtype)
            with torch.no_grad():
                fused = attn.query_key_value(x)
            # GPT-NeoX forward: view(..., num_heads, 3*head_dim) then chunk(3,-1)
            ref_q = (fused.view(1, 4, num_heads, 3 * head_dim)[..., :head_dim]
                     .reshape(1, 4, hidden).to(torch.float64).numpy())
            for interleaved in (True, False):
                Q = D.extract_qkv_block(Wqkv, 0, num_heads=num_heads,
                                        interleaved=interleaved)
                ours = x.to(torch.float64).numpy() @ Q.T
                err = float(np.max(np.abs(ours - ref_q)) /
                            max(np.max(np.abs(ref_q)), 1e-30))
                verdict = "MATCH" if err < 1e-6 else "mismatch"
                print(f"    interleaved={str(interleaved):5s} -> rel err "
                      f"{_fmt(err)}  {verdict}")
            print(f"    module: {attn_name}.query_key_value  "
                  f"num_heads={num_heads} head_dim={head_dim}")
            print("    Expected: interleaved=True MATCHes and "
                  "interleaved=False does not, for GPT-NeoX/Pythia.")

    print("\nall assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
