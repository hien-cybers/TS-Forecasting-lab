from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

for directory in [DATA_RAW_DIR, DATA_PROC_DIR, FIG_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATASET_URL = "https://raw.githubusercontent.com/selva86/datasets/master/a10.csv"
ALPHA = 0.05
TRAIN_RATIO = 0.80
ACF_PACF_LAGS = 36
candidate_orders = [(1, 1, 1), (2, 1, 1)]