import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 4)
    }

def run_arima_pipeline(df: pd.DataFrame, proc_dir: Path, fig_dir: Path, rep_dir: Path):
    print("\n" + "="*70)
    print("BÀI 5: THIẾT LẬP ARIMA / SARIMA - DỰ ĐOÁN - ĐÁNH GIÁ")
    print("="*70)
    
    n_total = len(df)
    n_train = int(n_total * 0.80)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    
    train_df.to_csv(proc_dir / "train_set.csv")
    test_df.to_csv(proc_dir / "test_set.csv")
    print(f"[+] Train set: {len(train_df)} quan sát | Test set: {len(test_df)} quan sát")
    
    models_spec = {
        "ARIMA(1,1,1)": {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0)},
        "ARIMA(2,1,1)": {"order": (2, 1, 1), "seasonal_order": (0, 0, 0, 0)},
        "SARIMA(1,1,1)(1,1,1,12)": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
    }
    
    fitted_models = {}
    ic_records = []
    metric_records = []
    
    for name, spec in models_spec.items():
        if spec["seasonal_order"] == (0, 0, 0, 0):
            model = ARIMA(train_df["value"], order=spec["order"])
        else:
            model = SARIMAX(train_df["value"], order=spec["order"], seasonal_order=spec["seasonal_order"])
            
        res = model.fit()
        fitted_models[name] = res
        
        converged = res.mle_retvals.get("converged", True)
        ic_records.append({
            "Model": name,
            "AIC": round(res.aic, 4),
            "BIC": round(res.bic, 4),
            "Log_Likelihood": round(res.llf, 4),
            "Converged": converged
        })
        
        fc = res.get_forecast(steps=len(test_df))
        y_pred = fc.predicted_mean.values
        metric_records.append(calculate_metrics(test_df["value"].values, y_pred, name))
        
    ic_df = pd.DataFrame(ic_records)
    metrics_df = pd.DataFrame(metric_records)
    ic_df.to_csv(rep_dir / "model_selection_aic_bic.csv", index=False)
    metrics_df.to_csv(rep_dir / "forecast_evaluation_metrics.csv", index=False)
    
    print("\n--- Bảng so sánh AIC / BIC trên tập Train ---")
    print(ic_df.to_string(index=False))
    
    # Chẩn đoán phần dư trên ARIMA(2,1,1)
    target_model_name = "ARIMA(2,1,1)"
    res_arima = fitted_models[target_model_name]
    residuals = res_arima.resid.iloc[1:]
    
    lb_test = acorr_ljungbox(residuals, lags=[1, 6, 12, 24], return_df=True)
    lb_test.to_csv(rep_dir / "ljung_box_residuals.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(residuals.index, residuals.values, color="#9467bd", linewidth=1.3)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Residuals theo thời gian - {target_model_name} (Mean = {residuals.mean():.4f})")
    axes[0].set_ylabel(r"Phần dư ($e_t$)")
    plot_acf(residuals, lags=36, ax=axes[1], title=f"ACF của Residuals - {target_model_name}")
    plt.suptitle("Bài 5.3: Chẩn đoán Phần dư (Residual Diagnostics)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_residual_diagnostics.png", dpi=300)
    plt.close()
    
    # Vẽ đồ thị dự báo
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train_df.index, train_df["value"], label="Train Set (80%)", color="#1f77b4", linewidth=1.8)
    ax.plot(test_df.index, test_df["value"], label="Test Set - Thực tế (20%)", color="black", linewidth=2.2)
    
    colors = {"ARIMA(1,1,1)": "#ff7f0e", "ARIMA(2,1,1)": "#d62728", "SARIMA(1,1,1)(1,1,1,12)": "#2ca02c"}
    styles = {"ARIMA(1,1,1)": "--", "ARIMA(2,1,1)": "-.", "SARIMA(1,1,1)(1,1,1,12)": "-"}
    
    for name, res in fitted_models.items():
        fc_obj = res.get_forecast(steps=len(test_df))
        fc_series = pd.Series(fc_obj.predicted_mean.values, index=test_df.index)
        ax.plot(fc_series.index, fc_series.values, label=f"Forecast - {name}", color=colors[name], linestyle=styles[name], linewidth=2)
        
    ax.set_title("Bài 5.4: Đối sánh Tập Train, Tập Test Thực tế và Đường Dự báo (Forecast)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian (Năm)")
    ax.set_ylabel("Doanh số (Triệu USD)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_train_test_forecast.png", dpi=300)
    plt.close()
    
    print("\n--- Bảng Đánh giá Mô hình trên tập Test (Out-of-Sample) ---")
    print(metrics_df.to_string(index=False))