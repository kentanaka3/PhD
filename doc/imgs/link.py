import os
import sys

BASE_PATH = "/Users/admin/Desktop/Monica/PhD/OGS"
DIRECTORIES = [
  "OGSBackup",
	"OGSEQTransformer_INSTANCE",
	"OGSEQTransformer_INSTANCE_TP0.2S0.2",
	"OGSEQTransformer_INSTANCE_TP0.3S0.3",
	"OGSEQTransformer_ORIGINAL",
	"OGSEQTransformer_SCEDC",
	"OGSEQTransformer_STEAD",
	"OGSPhaseNet_ORIGINAL",
	"OGSPhaseNet_SCEDC",
	"OGSPhaseNet_STEAD",
	"OGSPyOcto",
	"OGSPyOcto_TP0.2S0.2",
	"OGSPyOcto_TP0.3S0.3",
	"TP0.2S0.2",
	"TP0.3S0.3"
]

def main(filename):
  for directory in DIRECTORIES:
    source = os.path.join(BASE_PATH, directory, filename)
    os.system(f"ln {source} {directory}/")

if __name__ == "__main__": main(sys.argv[1])