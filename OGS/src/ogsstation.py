import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, sys.argv[1])
from ogsutils import waveforms

def main():
  output = Path(sys.argv[6]) if len(sys.argv) > 6 else Path(".")
  output.mkdir(parents=True, exist_ok=True)
  waveforms(
    Path(sys.argv[2]), Path(sys.argv[3]),
    datetime.strptime(sys.argv[4], '%Y%m%d'),
    datetime.strptime(sys.argv[5], '%Y%m%d'),
    output=output
  )

if __name__ == "__main__":  main()
