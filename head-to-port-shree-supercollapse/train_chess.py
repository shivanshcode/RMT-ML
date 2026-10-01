"""train_chess.py — transformer training with the full-weight checkpoint saver.

``train.py`` is NOT modified and NOT duplicated. This module installs
``checkpoint_chess`` under the name ``checkpoint`` before importing ``train``,
so ``train.py``'s ``import checkpoint as _ckpt`` (line ~255, inside
``train_and_evaluate``) resolves to the transformer saver instead of the MLP
one. Everything else --- the optimizer, the schedule, the eval loop, the CSV
writer --- is the original code, byte for byte.

WHY A SHIM AND NOT A COPY
-------------------------
The alternative is a 322-line duplicate of ``train.py`` with two lines changed.
That duplicate would diverge from the original the first time either is fixed,
and the divergence would be silent: both files run, both produce loss curves,
and only a careful diff would show that the chess ladder is no longer training
the way the MLP ladder does. Ten lines that cannot drift are worth more than
three hundred that can.

WHY IT IS SAFE
--------------
``sys.modules`` is populated before ``train`` is imported, so the substitution
is in place no matter when ``train`` resolves the name --- whether the import
stays inside the function or is later hoisted to module scope. The name
``checkpoint`` is not otherwise used in this process.

WHAT CHANGES IN THE CHECKPOINTS
-------------------------------
``checkpoint.py`` keeps a parameter only when its path contains "fc1" or "fc2".
On the MLP that is every weight matrix. On the transformer it silently discards
all four attention matrices per block --- 12 of 18, and the bulk of the
parameters. ``checkpoint_chess.py`` saves every 2-D Param: Q/K/V/O from
attention, U/D from the MLP, plus the embeddings and the readout.

USAGE
-----
Via ``main_chess.py`` (what ``run_chess_sweep.slurm`` calls):

    python main_chess.py -cn chess_local model.D=768 seed=0 wandb_tag=chess

Directly:

    import train_chess
    train_chess.train_and_evaluate(cfg)

The MLP ladder keeps using ``main.py`` -> ``train.py`` -> ``checkpoint.py``
untouched, so existing MLP checkpoints and every RMT result derived from them
are unaffected.
"""
from __future__ import annotations

import sys

import checkpoint_chess

# Must happen BEFORE `import train`.
sys.modules["checkpoint"] = checkpoint_chess

import train as _train  # noqa: E402  (deliberately after the substitution)

train_and_evaluate = _train.train_and_evaluate

__all__ = ["train_and_evaluate"]
