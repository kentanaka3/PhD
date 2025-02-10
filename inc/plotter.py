import matplotlib.pyplot as plt

def CMfig():
  pass

class CMfig():
  def __init__(self, weights):
    self.weights = weights
    self.fig, _ax = plt.subplots(1, len(weights), figsize=(10, 5))
    self.fig.suptitle("Confusion Matrix")
    self.fig.tight_layout()
    self.fig.subplots_adjust(top=0.88)
    self.ax = _ax.flatten()
    return self.fig, self.ax

