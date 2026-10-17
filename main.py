from pathlib import Path

# Import 5 module của Dataset 1 (a10.csv)
from src.ex1_eda import run_eda
from src.ex2_adf_raw import run_adf_raw
from src.ex3_differencing import run_differencing
from src.ex4_acf_pacf import run_acf_pacf
from src.ex5_arima_pipeline import run_arima_pipeline

# Import 5 module của Dataset 2 (airline-passengers.csv)
from src.sarima_ex1_eda import run_sarima_ex1_eda
from src.sarima_ex2_adf_raw import run_sarima_ex2_adf_raw
from src.sarima_ex3_differencing import run_sarima_ex3_differencing
from src.sarima_ex4_acf_pacf import run_sarima_ex4_acf_pacf
from src.sarima_ex5_pipeline import run_sarima_ex5_pipeline

BASE_DIR = Path(__file__).resolve().parent
DIRS = {
    "raw": BASE_DIR / "data" / "raw",
    "proc": BASE_DIR / "data" / "processed",
    "fig": BASE_DIR / "outputs" / "figures",
    "rep": BASE_DIR / "outputs" / "reports",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

DATASET1_URL = "https://raw.githubusercontent.com/selva86/datasets/master/a10.csv"

if __name__ == "__main__":
    # PHẦN 1: 5 BÀI TẬP THỰC HÀNH TRÊN DATASET 1 (a10.csv)
    df1 = run_eda(DATASET1_URL, DIRS["raw"], DIRS["fig"])
    adf_raw1 = run_adf_raw(df1)
    diff1 = run_differencing(df1, adf_raw1, DIRS["proc"], DIRS["fig"], DIRS["rep"])
    run_acf_pacf(diff1, DIRS["fig"])
    run_arima_pipeline(df1, DIRS["proc"], DIRS["fig"], DIRS["rep"])

    # PHẦN 2: 5 BÀI TẬP NÂNG CAO VỚI SARIMA TRÊN DATASET 2 (airline-passengers.csv)
    df2 = run_sarima_ex1_eda(DIRS["raw"], DIRS["fig"])
    adf_raw2 = run_sarima_ex2_adf_raw(df2)
    diff2_comb = run_sarima_ex3_differencing(
        df2, adf_raw2, DIRS["proc"], DIRS["fig"], DIRS["rep"]
    )
    run_sarima_ex4_acf_pacf(diff2_comb, DIRS["fig"])
    run_sarima_ex5_pipeline(df2, DIRS["proc"], DIRS["fig"], DIRS["rep"])

    print(
        f"\n[HOÀN TẤT TOÀN BỘ LAB 2] Kiểm tra 10 biểu đồ & 6 báo cáo tại: {BASE_DIR / 'outputs'}"
    )