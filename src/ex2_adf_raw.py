import pandas as pd
from statsmodels.tsa.stattools import adfuller

def perform_adf_test(series: pd.Series, series_name: str, alpha: float = 0.05) -> dict:
    clean_series = series.dropna()
    result = adfuller(clean_series, autolag="AIC", result_object=False)
    adf_stat, p_val, used_lag, n_obs, crit_vals, _ = result
    is_stationary = (p_val < alpha) and (adf_stat < crit_vals["5%"])
    
    summary = {
        "Series": series_name,
        "ADF_Statistic": round(adf_stat, 6),
        "p_value": round(p_val, 6),
        "Used_Lags": used_lag,
        "N_Obs": n_obs,
        "Critical_1%": round(crit_vals["1%"], 4),
        "Critical_5%": round(crit_vals["5%"], 4),
        "Critical_10%": round(crit_vals["10%"], 4),
        "Conclusion": "Dừng (Stationary)" if is_stationary else "Không dừng (Non-Stationary)"
    }
    
    print(f"\n--- Kết quả Kiểm định ADF: {series_name} ---")
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")
    return summary

def run_adf_raw(df: pd.DataFrame) -> dict:
    print("\n" + "="*70)
    print("BÀI 2: KIỂM ĐỊNH ADF TRÊN CHUỖI GỐC")
    print("="*70)
    return perform_adf_test(df["value"], "Chuỗi gốc Y_t")