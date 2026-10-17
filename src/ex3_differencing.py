import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from src.ex2_adf_raw import perform_adf_test

def run_differencing(df: pd.DataFrame, adf_raw_res: dict, proc_dir: Path, fig_dir: Path, rep_dir: Path) -> pd.Series:
    print("\n" + "="*70)
    print("BÀI 3: SAI PHÂN BẬC 1 & KIỂM ĐỊNH LẠI ADF")
    print("="*70)
    
    diff1 = df["value"].diff(1).dropna()
    diff1.to_csv(proc_dir / "a10_diff1.csv")
    
    adf_diff1_aic = perform_adf_test(diff1, "Sai phân bậc 1 (Delta Y_t - autolag='AIC')")
    log_diff1 = np.log(df["value"]).diff(1).dropna()
    adf_log_diff1 = perform_adf_test(log_diff1, "Sai phân bậc 1 trên chuỗi Log (Delta ln(Y_t))")
    
    res_short = adfuller(diff1, maxlag=1, autolag=None, result_object=False)
    adf_diff1_short = {
        "Series": "Sai phân bậc 1 (Delta Y_t - maxlag=1)",
        "ADF_Statistic": round(res_short[0], 6),
        "p_value": round(res_short[1], 6),
        "Used_Lags": res_short[2],
        "N_Obs": res_short[3],
        "Critical_1%": round(res_short[4]["1%"], 4),
        "Critical_5%": round(res_short[4]["5%"], 4),
        "Critical_10%": round(res_short[4]["10%"], 4),
        "Conclusion": "Dừng (Stationary)" if res_short[1] < 0.05 else "Không dừng (Non-Stationary)"
    }
    
    pd.DataFrame([adf_raw_res, adf_diff1_aic, adf_diff1_short, adf_log_diff1]).to_csv(
        rep_dir / "adf_test_results.csv", index=False
    )
    
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(diff1.index, diff1.values, color="#ff7f0e", linewidth=1.4, label=r"$\Delta Y_t = Y_t - Y_{t-1}$")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="Mean = 0")
    ax.set_title(r"Bài 3: Đồ thị Chuỗi thời gian sau Sai phân bậc 1 ($d=1$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian (Năm)")
    ax.set_ylabel(r"$\Delta Y_t$")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_diff1_series.png", dpi=300)
    plt.close()
    
    return diff1