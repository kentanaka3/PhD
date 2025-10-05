import os
import sys
import argparse

import ogsconstants as OGS_C

def parse_arguments():
  parser = argparse.ArgumentParser(description="Train OGS models")
  parser.add_argument("-c", "--config", type=str, required=True,
                      help="Path to the configuration file")
  parser.add_argument("-d", "--debug", action="store_true",
                      help="Enable debug mode")
  return parser.parse_args()

class OGSTrainer:
  def __init__(self, args):
    self.config_path = args.config
    self.debug = args.debug
    self.load_config()

  def load_config(self):
    if not os.path.exists(self.config_path):
      print(f"Configuration file {self.config_path} does not exist.")
      sys.exit(1)
    # Load configuration logic here
    print(f"Loaded configuration from {self.config_path}")

  def train(self):
    # Training logic here
    if self.debug:
      print("Debug mode is enabled.")
    print("Training OGS models...")

def main(args):
  trainer = OGSTrainer(args)
  trainer.train()


if __name__ == "__main__": main(parse_arguments())