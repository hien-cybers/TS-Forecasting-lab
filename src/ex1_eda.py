import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_eda(dataset_url: str, raw_dir: Path, fig_dir: Path) -> pd.DataFrame:
    print("\n" + "="*70)
    print("BÀI 1: KHÁM PHÁ DỮ LIỆU (EDA)")
    print("="*70)
    
    df = pd.read_csv(dataset_url)
    df.to_csv(raw_dir / "a10_raw.csv", index=False)
    
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.index.freq = "MS"
    
    print(f"[+] Kích thước dữ liệu: {df.shape[0]} quan sát ({df.index.min().strftime('%Y-%m')} -> {df.index.max().strftime('%Y-%m')})")
    print(f"[+] Thống kê mô tả:\n{df['value'].describe().round(4)}")
    
    rolling_mean = df["value"].rolling(window=12).mean()
    rolling_std = df["value"].rolling(window=12).std()
    
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df.index, df["value"], label=r"Chuỗi gốc $Y_t$ (Monthly Sales)", color="#1f77b4", linewidth=1.8)
    ax.plot(df.index, rolling_mean, label="Rolling Mean (12M) - Xu hướng", color="#d62728", linestyle="--", linewidth=2)
    ax.plot(df.index, rolling_std, label="Rolling Std (12M) - Phương sai", color="#2ca02c", linestyle=":", linewidth=2)
    ax.set_title("Bài 1: Đồ thị Chuỗi thời gian gốc Dataset 1 (a10.csv)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian (Năm)")
    ax.set_ylabel("Doanh số (Triệu USD)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_eda_original_series.png", dpi=300)
    plt.close()
    
    return df