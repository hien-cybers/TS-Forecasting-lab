import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def run_acf_pacf(diff1: pd.Series, fig_dir: Path):
    print("\n" + "="*70)
    print("BÀI 4: PHÂN TÍCH ACF & PACF CỦA CHUỖI ĐÃ SAI PHÂN")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(diff1, lags=36, ax=axes[0], title=r"Hàm Tự tương quan (ACF) của $\Delta Y_t$")
    plot_pacf(diff1, lags=36, ax=axes[1], method="ywm", title=r"Hàm Tự tương quan Riêng phần (PACF) của $\Delta Y_t$")
    plt.suptitle("Bài 4: Phân tích ACF & PACF để xác định tham số (p, d, q)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_acf_pacf_diff1.png", dpi=300)
    plt.close()
    print(f"[+] Đã xuất lược đồ ACF/PACF tại: {fig_dir / 'fig3_acf_pacf_diff1.png'}")