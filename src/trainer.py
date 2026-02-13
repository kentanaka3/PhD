# python src/trainer.py --file ./data/manual/ -v -D 230601 230609 -W OGS
import argparse
import seisbench.data as sbd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
INC_PATH = os.path.join(PRJ_PATH, "inc")
IMG_PATH = os.path.join(PRJ_PATH, "img")
DATA_PATH = os.path.join(PRJ_PATH, "data")
# Add to path
if INC_PATH not in sys.path:
  sys.path.append(INC_PATH)
  from resources.constants import *
  import initializer as ini


# Seisbench


class TrainConfig:
  def __init__(self, SOURCE: pd.DataFrame, DETECT: pd.DataFrame,
               WAVEFORMS: pd.DataFrame, args: argparse.Namespace):
    self.source = SOURCE
    self.detect = DETECT
    self.waveforms = WAVEFORMS
    self.args = args


def dataset_loader(args: argparse.Namespace):
  assert len(args.weights) == 1
  WEIGHT = args.weights[0]
  DATASET_PATH = Path(DATA_PATH, MODELS_STR, WEIGHT)
  if args.force or (not DATASET_PATH.exists()):
    print(f"Creating dataset for {WEIGHT}")
    ini.dataset_builder(args)
  DATA = sbd.WaveformDataset(DATASET_PATH, WEIGHT)
  if args.verbose and args.force:
    w, m = DATA.get_sample(int(np.random.random() * len(DATA)))
  global IMG_PATH
  plt.tight_layout()
  plt.savefig(Path(IMG_PATH, WEIGHT + PNG_EXT), bbox_inches='tight')
  plt.close()


def main(args: argparse.Namespace):
  dataset_loader(args)


if __name__ == "__main__":
  main(ini.parse_arguments())
