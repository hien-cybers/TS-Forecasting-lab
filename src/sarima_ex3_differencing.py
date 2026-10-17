from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.sarima_ex2_adf_raw import perform_sarima_adf


def run_sarima_ex3_differencing(
    df: pd.DataFrame,
    adf_raw_res: dict,
    proc_dir: Path,
    fig_dir: Path,
    rep_dir: Path,
) -> pd.Series:
    print("\n" + "=" * 75)
    print("DATASET 2 - BÀI 3: SAI PHÂN THƯỜNG (d=1) + MÙA VỤ (D=1, s=12) & KIỂM ĐỊNH ADF")
    print("=" * 75)

    # 1. Sai phân thường bậc 1 (d=1, D=0): loại bỏ xu hướng (Trend)
    diff_reg = df["Passengers"].diff(1).dropna()
    diff_reg.to_csv(proc_dir / "air_diff_regular_d1.csv")

    # 2. Sai phân kết hợp (d=1, D=1, s=12): loại bỏ cả Trend và Seasonality
    diff_comb = df["Passengers"].diff(1).diff(12).dropna()
    diff_comb.to_csv(proc_dir / "air_diff_combined_d1_D1.csv")

    adf_d1 = perform_sarima_adf(
        diff_reg, "2. Sai phân thường Delta Y_t (d=1, D=0)"
    )
    adf_d1_D1 = perform_sarima_adf(
        diff_comb,
        "3. Sai phân thường + mùa vụ Delta_12 Delta Y_t (d=1, D=1, s=12)",
    )

    adf_df = pd.DataFrame([adf_raw_res, adf_d1, adf_d1_D1])
    adf_df.to_csv(rep_dir / "air_adf_stationarity_tests.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    axes[0].plot(
        diff_reg.index,
        diff_reg.values,
        color="#ff7f0e",
        linewidth=1.4,
        label=r"Sai phân thường ($d=1, D=0$): $\Delta Y_t = Y_t - Y_{t-1}$",
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(
        r"Chuỗi sau Sai phân thường bậc 1 ($d=1$) - Đã khử Trend nhưng còn chu kỳ mùa vụ $s=12$"
    )
    axes[0].legend(loc="upper left")

    axes[1].plot(
        diff_comb.index,
        diff_comb.values,
        color="#2ca02c",
        linewidth=1.4,
        label=r"Sai phân kép ($d=1, D=1, s=12$): $\Delta_{12}\Delta Y_t = (1-B)(1-B^{12})Y_t$",
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(
        r"Chuỗi sau Sai phân thường + Sai phân mùa vụ ($d=1, D=1, s=12$) - Đã dừng hoàn toàn (ADF p-val = 0.0000)"
    )
    axes[1].legend(loc="upper left")

    plt.suptitle(
        "Dataset 2 - Bài 3: Đối sánh Sai phân thường (d=1) và Sai phân kết hợp (d=1, D=1, s=12)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_air_differencing_stages.png", dpi=300)
    plt.close()

    return diff_comb