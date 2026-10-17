from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_sarima_ex5_pipeline(
    df: pd.DataFrame, proc_dir: Path, fig_dir: Path, rep_dir: Path
) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("DATASET 2 - BÀI 5: FIT SARIMA - DỰ BÁO - ĐÁNH GIÁ & SO SÁNH VỚI ARIMA")
    print("=" * 75)

    n_train = int(len(df) * 0.80)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    train_df.to_csv(proc_dir / "air_train_set.csv")
    test_df.to_csv(proc_dir / "air_test_set.csv")
    print(
        f"[+] Train set: {len(train_df)} quan sát ({train_df.index.min().strftime('%Y-%m')} -> {train_df.index.max().strftime('%Y-%m')})"
    )
    print(
        f"[+] Test set : {len(test_df)} quan sát ({test_df.index.min().strftime('%Y-%m')} -> {test_df.index.max().strftime('%Y-%m')})"
    )

    models_spec = {
        "ARIMA(1,1,1) [Non-Seasonal]": {
            "order": (1, 1, 1),
            "seasonal_order": (0, 0, 0, 0),
        },
        "ARIMA(2,1,1) [Non-Seasonal]": {
            "order": (2, 1, 1),
            "seasonal_order": (0, 0, 0, 0),
        },
        "SARIMA(0,1,1)(0,1,1,12) [Airline Model]": {
            "order": (0, 1, 1),
            "seasonal_order": (0, 1, 1, 12),
        },
        "SARIMA(1,1,1)(1,1,1,12) [Full SARIMA]": {
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 12),
        },
    }

    fitted_models = {}
    comparison_rows = []
    forecasts = {}
    conf_ints = {}
    y_true = test_df["Passengers"].values

    for name, spec in models_spec.items():
        if spec["seasonal_order"] == (0, 0, 0, 0):
            model = ARIMA(train_df["Passengers"], order=spec["order"])
        else:
            model = SARIMAX(
                train_df["Passengers"],
                order=spec["order"],
                seasonal_order=spec["seasonal_order"],
            )

        res = model.fit()
        fitted_models[name] = res

        fc_obj = res.get_forecast(steps=len(test_df))
        y_pred = fc_obj.predicted_mean.values
        forecasts[name] = pd.Series(y_pred, index=test_df.index)
        conf_ints[name] = fc_obj.conf_int(alpha=0.05)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        resid = res.resid.iloc[13:]
        lb_pval_12 = acorr_ljungbox(resid, lags=[12], return_df=True).loc[
            12, "lb_pvalue"
        ]

        comparison_rows.append(
            {
                "Model": name,
                "AIC (Train)": round(res.aic, 4),
                "BIC (Train)": round(res.bic, 4),
                "MAE (Test)": round(mae, 4),
                "RMSE (Test)": round(rmse, 4),
                "MAPE (Test %)": round(mape, 4),
                "Ljung-Box p-val (Lag 12)": round(lb_pval_12, 6),
                "Converged": res.mle_retvals.get("converged", True),
            }
        )

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(rep_dir / "air_sarima_vs_arima_comparison.csv", index=False)

    print("\n--- Bảng Đối sánh Toàn diện SARIMA vs ARIMA trên Dataset 2 ---")
    print(comp_df.to_string(index=False))

    # Vẽ Chẩn đoán phần dư đối sánh ARIMA vs SARIMA
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    check_pair = [
        "ARIMA(2,1,1) [Non-Seasonal]",
        "SARIMA(0,1,1)(0,1,1,12) [Airline Model]",
    ]
    for idx, m_name in enumerate(check_pair):
        resid = fitted_models[m_name].resid.iloc[13:]
        axes[idx, 0].plot(resid.index, resid.values, color="#9467bd", linewidth=1.3)
        axes[idx, 0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[idx, 0].set_title(
            f"Residuals: {m_name} (Mean = {resid.mean():.3f})"
        )
        plot_acf(resid, lags=36, ax=axes[idx, 1], title=f"ACF Residuals: {m_name}")

    plt.suptitle(
        "Dataset 2 - Bài 5.1: Chẩn đoán Phần dư đối sánh ARIMA vs SARIMA",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_air_sarima_residuals.png", dpi=300)
    plt.close()

    # Vẽ đồ thị Dự báo (Forecast) đối sánh ARIMA và SARIMA
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        train_df.index,
        train_df["Passengers"],
        label="Train Set (80% - 115 tháng)",
        color="#1f77b4",
        linewidth=1.8,
    )
    ax.plot(
        test_df.index,
        test_df["Passengers"],
        label="Test Set - Thực tế (20% - 29 tháng)",
        color="black",
        linewidth=2.3,
    )

    palette = {
        "ARIMA(1,1,1) [Non-Seasonal]": ("#ff7f0e", "--"),
        "ARIMA(2,1,1) [Non-Seasonal]": ("#d62728", "-."),
        "SARIMA(0,1,1)(0,1,1,12) [Airline Model]": ("#2ca02c", "-"),
        "SARIMA(1,1,1)(1,1,1,12) [Full SARIMA]": ("#9467bd", ":"),
    }

    for name, fc_s in forecasts.items():
        color, style = palette[name]
        ax.plot(
            fc_s.index,
            fc_s.values,
            label=f"Forecast: {name}",
            color=color,
            linestyle=style,
            linewidth=2.1,
        )

    best_ci = conf_ints["SARIMA(0,1,1)(0,1,1,12) [Airline Model]"]
    ax.fill_between(
        test_df.index,
        best_ci.iloc[:, 0],
        best_ci.iloc[:, 1],
        color="#2ca02c",
        alpha=0.15,
        label="95% CI - SARIMA(0,1,1)(0,1,1,12)",
    )

    ax.set_title(
        "Dataset 2 - Bài 5.2: Đối sánh Dự báo SARIMA vs ARIMA (AirPassengers)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Thời gian (Năm)")
    ax.set_ylabel("Số lượng hành khách (Nghìn người)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig10_air_arima_vs_sarima_forecast.png", dpi=300)
    plt.close()

    return comp_df