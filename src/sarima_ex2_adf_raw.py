import pandas as pd
from statsmodels.tsa.stattools import adfuller


def perform_sarima_adf(
    series: pd.Series, stage_name: str, alpha: float = 0.05
) -> dict:
    clean_s = series.dropna()
    stat, pval, usedlag, nobs, crits, _ = adfuller(
        clean_s, autolag="AIC", result_object=False
    )
    is_stat = (pval < alpha) and (stat < crits["5%"])
    summary = {
        "Stage": stage_name,
        "ADF_Statistic": round(stat, 6),
        "p_value": round(pval, 6),
        "Used_Lags": usedlag,
        "N_Obs": nobs,
        "Critical_1%": round(crits["1%"], 4),
        "Critical_5%": round(crits["5%"], 4),
        "Critical_10%": round(crits["10%"], 4),
        "Conclusion": "Dừng (Stationary)" if is_stat else "Không dừng (Non-Stationary)",
    }

    print(f"\n--- Kết quả Kiểm định ADF: {stage_name} ---")
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")
    return summary


def run_sarima_ex2_adf_raw(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 75)
    print("DATASET 2 - BÀI 2: KIỂM ĐỊNH NGHIỆM ĐƠN VỊ ADF TRÊN CHUỖI GỐC")
    print("=" * 75)
    return perform_sarima_adf(df["Passengers"], "1. Chuỗi gốc Y_t (d=0, D=0)")