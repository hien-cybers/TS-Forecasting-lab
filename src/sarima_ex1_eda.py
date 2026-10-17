from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

AIRLINE_URL_PRIMARY = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
AIRLINE_URL_FALLBACK = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airlinepassengers.csv"


def run_sarima_ex1_eda(raw_dir: Path, fig_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("DATASET 2 - BÀI 1: KHÁM PHÁ DỮ LIỆU & NHẬN DIỆN MÙA VỤ (AIRPASSENGERS)")
    print("=" * 75)

    try:
        df = pd.read_csv(AIRLINE_URL_PRIMARY)
    except Exception:
        df = pd.read_csv(AIRLINE_URL_FALLBACK)

    df.to_csv(raw_dir / "airline_passengers_raw.csv", index=False)
    df.columns = ["Month", "Passengers"]
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)
    df.index.freq = "MS"

    print(
        f"[+] Kích thước dữ liệu: {df.shape[0]} quan sát ({df.index.min().strftime('%Y-%m')} -> {df.index.max().strftime('%Y-%m')})"
    )
    print(f"[+] Thống kê mô tả:\n{df['Passengers'].describe().round(4)}")

    rolling_mean = df["Passengers"].rolling(window=12).mean()
    rolling_std = df["Passengers"].rolling(window=12).std()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(
        df.index,
        df["Passengers"],
        color="#1f77b4",
        linewidth=1.8,
        label=r"Hành khách thực tế ($Y_t$)",
    )
    axes[0].plot(
        df.index,
        rolling_mean,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label="Rolling Mean (12M) - Trend",
    )
    axes[0].plot(
        df.index,
        rolling_std,
        color="#2ca02c",
        linestyle=":",
        linewidth=2,
        label="Rolling Std (12M) - Phương sai",
    )
    axes[0].set_title(
        "1A. Chuỗi thời gian AirPassengers (1949 - 1960)", fontweight="bold"
    )
    axes[0].set_xlabel("Thời gian (Năm)")
    axes[0].set_ylabel("Số lượng hành khách (Nghìn người)")
    axes[0].legend(loc="upper left")

    df_pivot = df.copy()
    df_pivot["Month_Num"] = df_pivot.index.month
    monthly_means = df_pivot.groupby("Month_Num")["Passengers"].mean()
    axes[1].plot(
        range(1, 13),
        monthly_means.values,
        marker="o",
        color="#ff7f0e",
        linewidth=2.2,
    )
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title(
        "1B. Cấu trúc mùa vụ trung bình theo Tháng (Chu kỳ s = 12)",
        fontweight="bold",
    )
    axes[1].set_xlabel("Tháng trong năm (1 -> 12)")
    axes[1].set_ylabel("Trung bình hành khách (Nghìn người)")

    plt.suptitle(
        "Dataset 2 - Bài 1: Nhận diện Xu hướng (Trend) và Mùa vụ (Seasonality s=12)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_air_eda_seasonality.png", dpi=300)
    plt.close()

    return df