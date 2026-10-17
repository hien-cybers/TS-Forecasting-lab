from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def run_sarima_ex4_acf_pacf(diff_comb: pd.Series, fig_dir: Path):
    print("\n" + "=" * 75)
    print("DATASET 2 - BÀI 4: PHÂN TÍCH ACF & PACF XÁC ĐỊNH (p, q, P, Q)")
    print("=" * 75)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(
        diff_comb,
        lags=36,
        ax=axes[0],
        title=r"ACF của chuỗi $\Delta_{12}\Delta Y_t$ ($d=1, D=1, s=12$)",
    )
    plot_pacf(
        diff_comb,
        lags=36,
        ax=axes[1],
        method="ywm",
        title=r"PACF của chuỗi $\Delta_{12}\Delta Y_t$ ($d=1, D=1, s=12$)",
    )
    plt.suptitle(
        "Dataset 2 - Bài 4: Lược đồ ACF & PACF để xác định tham số (p, q) và (P, Q)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_air_acf_pacf_d1_D1.png", dpi=300)
    plt.close()
    print(
        f"[+] Đã xuất lược đồ ACF/PACF của Dataset 2 tại: {fig_dir / 'fig8_air_acf_pacf_d1_D1.png'}"
    )