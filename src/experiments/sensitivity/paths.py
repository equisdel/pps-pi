from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"

X_INPUT_PATH = str(DATA_DIR / "X.txt")
Y_OUTPUT_PATH = 'experiments/sensitivity/HV_final_jpetstore_original.csv' #str(DATA_DIR / "Y.csv")
PF_OUTPUT_PATH = str(DATA_DIR / "PF.json")
HV_GEN_OUTPUT_PATH = str(DATA_DIR / "HV_by_generation.csv")
