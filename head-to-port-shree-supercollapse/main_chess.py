"""main_chess.py — hydra entry point for the chess transformer sweep.

Identical to ``main.py`` except that it calls ``train_chess`` instead of
``train``, which routes weight checkpointing through ``checkpoint_chess`` (all
18 block matrices, not just the 6 MLP ones). ``main.py`` and ``train.py`` are
untouched and still drive the MLP ladder.

    python main_chess.py -cn chess_local model.D=768 seed=0 wandb_tag=chess

The default config is ``chess_local`` rather than ``local``, so an omitted
``-cn`` does not silently launch the MLP config on the transformer path.
"""
import hydra
import train_chess
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='configs', config_name='chess_local')
def main(c: DictConfig):
    train_chess.train_and_evaluate(c)


if __name__ == '__main__':
    main()
