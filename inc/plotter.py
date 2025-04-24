import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
# Set the project folder
PRJ_PATH = Path(os.path.dirname(__file__)).parent
IMG_PATH = Path(PRJ_PATH, "img")
DATA_PATH = Path(PRJ_PATH, "data")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def CMfig():
  pass

class CMfig():
  def __init__(self, weights, dates : list[str] = None):
    self.weights = weights
    if dates is not None:
      self.dates = dates
    else:
      pass
    self.fig, _ax = plt.subplots(1, len(weights), figsize=(10, 5))
    self.fig.suptitle("Confusion Matrix")
    self.fig.tight_layout()
    self.fig.subplots_adjust(top=0.88)
    self.ax = _ax.flatten()
    return self.fig, self.ax

