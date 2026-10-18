# 📈 Time Series Forecasting Lab: End-to-End ARIMA & SARIMA Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Pipeline-Automated-brightgreen)

A research and hands-on project focused on **Time Series Modeling and Forecasting** using the **Box–Jenkins methodology**. The source code follows a **Modular Data Science Pipeline** architecture, clearly separating configuration, data, processing code, automated orchestration scripts, and visual reports.

---

## 🎯 Key Highlights

- **Modular Architecture:** Each stage of the analysis is independently organized: EDA → ADF unit-root testing → trend/seasonal differencing → ACF/PACF order identification → MLE estimation, residual diagnostics, and forecasting.
- **End-to-End Automation:** With a single command via `main.py`, the system automatically downloads data, preprocesses it, trains models, and exports 10 high-quality figures (`300 DPI`) and 6 `.csv` report tables.
- **Multi-Dataset Benchmarking:**
  1. **Dataset 1 (`a10.csv`):** Australian monthly retail sales of antidiabetic drugs (`N = 204` months) — basic analysis and ARIMA vs. SARIMA comparison.
  2. **Dataset 2 (`airline-passengers.csv`):** International airline passenger traffic (`N = 144` months) — an advanced forecasting task involving increasing variance and strong seasonality (`s = 12`).

---

## 🗂️ Project Structure

```text
Lab2_Timeseries/
├── config/
│   └── settings.py                          # Hyperparameters, data URLs, and train/test split
├── data/
│   ├── raw/                                 # Original raw data (a10_raw.csv, airline_passengers_raw.csv)
│   └── processed/                           # Differenced data (d=1, D=1) and train/test sets (80/20)
├── notebooks/
│   ├── Lab2_Walkthrough.ipynb               # Detailed interactive notebook for Dataset 1 (a10.csv)
│   └── Lab2_Advanced_SARIMA.ipynb           # In-depth 7-step SARIMA notebook for Dataset 2 (AirPassengers)
├── outputs/
│   ├── figures/                             # 10 analysis, residual diagnostic, and forecast figures (.png)
│   └── reports/                             # 6 report tables for ADF, AIC/BIC, Ljung-Box, and metrics (.csv)
├── src/
│   ├── __init__.py                          # Python package initialization
│   ├── ex1_eda.py                           # [Dataset 1] Exercise 1: Data exploration and EDA
│   ├── ex2_adf_raw.py                       # [Dataset 1] Exercise 2: ADF test on the original series
│   ├── ex3_differencing.py                  # [Dataset 1] Exercise 3: First-order differencing (d=1) and repeat ADF test
│   ├── ex4_acf_pacf.py                      # [Dataset 1] Exercise 4: ACF and PACF analysis
│   ├── ex5_arima_pipeline.py                # [Dataset 1] Exercise 5: ARIMA/SARIMA training, diagnostics, and evaluation
│   ├── sarima_ex1_eda.py                    # [Dataset 2] Exercise 1: Plot the time series and identify seasonality (s=12)
│   ├── sarima_ex2_adf_raw.py                # [Dataset 2] Exercise 2: ADF test on the original series → non-stationary
│   ├── sarima_ex3_differencing.py           # [Dataset 2] Exercise 3: Regular differencing (d=1) + seasonal differencing (D=1, s=12)
│   ├── sarima_ex4_acf_pacf.py               # [Dataset 2] Exercise 4: ACF/PACF analysis → identify (p, q, P, Q)
│   └── sarima_ex5_pipeline.py               # [Dataset 2] Exercise 5: Fit SARIMA, forecast, evaluate, and compare with ARIMA
├── generate_notebook.py                     # Script to automatically generate the Dataset 1 Jupyter Notebook
├── generate_sarima_notebook.py               # Script to automatically generate the Dataset 2 Jupyter Notebook
├── main.py                                  # Orchestrates the entire pipeline from start to finish
├── requirements.txt                         # Required Python packages
└── README.md                                # Project overview and documentation
```

---

## 📐 Methodology & Mathematical Foundations

### 1. Differencing Operators for Trend and Seasonality

- **First-order regular differencing, `d = 1` (removes a linear trend):**

  $$
  \Delta Y_t = (1 - B)Y_t = Y_t - Y_{t-1}
  $$

- **Combined regular differencing `d = 1` and seasonal differencing `D = 1` with period `s = 12` (removes both trend and seasonality):**

  $$
  \Delta_{12}\Delta Y_t = (1 - B)(1 - B^{12})Y_t = Y_t - Y_{t-1} - Y_{t-12} + Y_{t-13}
  $$

### 2. General SARIMA Model: `SARIMA(p, d, q)(P, D, Q)_s`

The lag-polynomial equation for a seasonal autoregressive integrated moving-average model is:

$$
\phi_p(B)\Phi_P(B^s)(1 - B)^d(1 - B^s)^D Y_t
= \theta_q(B)\Theta_Q(B^s)\varepsilon_t,
\quad \varepsilon_t \sim \text{WN}(0, \sigma^2)
$$

### 3. Out-of-Time Forecast Evaluation Metrics

For a test set containing `n` future observations, forecast quality is measured by comparing predicted values `Ŷ_t` with actual values `Y_t` using the following metrics:

$$
\mathrm{MAE} = \frac{1}{n}\sum_{t=1}^{n} \left|Y_t - \hat{Y}_t\right|
$$

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n} \left(Y_t - \hat{Y}_t\right)^2}
$$

$$
\mathrm{MAPE} = \frac{100\%}{n}\sum_{t=1}^{n}
\left|\frac{Y_t - \hat{Y}_t}{Y_t}\right|
$$

---

## 📊 Experimental Results & Visualizations

### Part 1: Retail Sales Forecasting (`a10.csv`)

#### 1.1. Exploratory Data Analysis, Differencing, and Correlation (ACF/PACF)

| Original Series & Rolling Mean/Std (Exercise 1) | First-Order Differenced Series, `d=1` (Exercise 3) |
| :---: | :---: |
| ![EDA a10](outputs/figures/fig1_eda_original_series.png) | ![Diff1 a10](outputs/figures/fig2_diff1_series.png) |

![ACF PACF a10](outputs/figures/fig3_acf_pacf_diff1.png)

#### 1.2. Model Comparison on Dataset 1 (`a10.csv`)

| Model | AIC (Train) | BIC (Train) | MAE (Test) | RMSE (Test) | MAPE (Test %) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **ARIMA(1, 1, 1)** | 727.4981 | 736.7510 | 3.7126 | 4.6852 | 18.2104% |
| **ARIMA(2, 1, 1)** | 722.4015 | 734.7387 | 3.6702 | 4.6285 | 18.0331% |
| **SARIMA(1, 1, 1)(1, 1, 1)₁₂** | **454.8124** | **469.7541** | **1.5608** | **2.0023** | **9.0630%** |

![Forecast a10](outputs/figures/fig5_train_test_forecast.png)

---

### Part 2: Advanced SARIMA Exercise on Dataset 2 (`airline-passengers.csv`)

#### 2.1. Identifying Seasonality (`s = 12`) and Testing Stationarity at Different Differencing Stages

![EDA AirPassengers](outputs/figures/fig6_air_eda_seasonality.png)

| Series Processing Stage | Operator | ADF Statistic | p-value | Critical Value (5%) | Conclusion (`α = 0.05`) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **1. Original series `Y_t` (`d=0, D=0`)** | `Y_t` | +0.815369 | 0.991880 | -2.8840 | **Non-stationary** |
| **2. Regular differencing (`d=1, D=0`)** | `ΔY_t = (1-B)Y_t` | -2.829267 | 0.054213 | -2.8840 | **Non-stationary** (seasonality remains) |
| **3. Double differencing (`d=1, D=1, s=12`)** | `Δ₁₂ΔY_t = (1-B)(1-B¹²)Y_t` | **-15.595618** | **0.000000** | -2.8842 | **Stationary** |

| Regular vs. Double Differencing (Step 3) | ACF/PACF Plots for Identifying `(p, q, P, Q)` (Step 4) |
| :---: | :---: |
| ![Differencing AirPassengers](outputs/figures/fig7_air_differencing_stages.png) | ![ACF PACF AirPassengers](outputs/figures/fig8_air_acf_pacf_d1_D1.png) |

#### 2.2. Forecast Performance Comparison: SARIMA vs. ARIMA on Dataset 2

| Model | AIC (Train) | BIC (Train) | Ljung-Box `p-value` (Lag 12) | MAE (Test) | RMSE (Test) | MAPE (Test %) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ARIMA(1, 1, 1)** *(non-seasonal)* | 1096.0689 | 1104.2775 | 0.000000 *(white noise rejected)* | 68.5772 | 92.0528 | 14.6811% |
| **ARIMA(2, 1, 1)** *(non-seasonal)* | 1082.7538 | 1093.6986 | 0.000000 *(white noise rejected)* | 66.2415 | 88.3614 | 14.2796% |
| **SARIMA(0, 1, 1)(0, 1, 1)₁₂** *(Airline Model)* | **754.2289** | **762.1319** | **0.728941** *(consistent with white noise)* | **17.7981** | **21.8385** | **4.0182%** |
| **SARIMA(1, 1, 1)(1, 1, 1)₁₂** *(Full SARIMA)* | 756.0125 | 769.1842 | 0.714520 *(consistent with white noise)* | 18.1406 | 22.2519 | 4.0954% |

| ARIMA vs. SARIMA Residual Diagnostics (Step 5) | Out-of-Sample Forecast Plot (Steps 6 & 7) |
| :---: | :---: |
| ![AirPassengers Residuals](outputs/figures/fig9_air_sarima_residuals.png) | ![AirPassengers Forecast](outputs/figures/fig10_air_arima_vs_sarima_forecast.png) |

> **Key finding:** The **SARIMA(0,1,1)(0,1,1)₁₂** model captures both the upward trend and the annual seasonal peaks and troughs effectively. It reduces absolute percentage forecast error by more than a factor of 3.8 compared with conventional ARIMA, lowering **MAPE from 14.68% to just 4.02%**.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/hien-cybers/ts-forecasting-lab
cd Lab2_Timeseries
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Entire Automated Pipeline

Run the central orchestration script to regenerate all data, figures, and reports for both datasets:

```bash
python main.py
```

### 4. Explore the Jupyter Notebooks (Optional)

You can open the two interactive notebooks in the `notebooks/` directory:

- `notebooks/Lab2_Walkthrough.ipynb` — 35 detailed cells for Dataset 1.
- `notebooks/Lab2_Advanced_SARIMA.ipynb` — 26 detailed cells for Dataset 2.

---

## 👤 Author

- **GitHub Profile:** [@hien-cybers](https://github.com/hien-cybers)
- **Repository:** [ts-forecasting-lab](https://github.com/hien-cybers/TS-Forecasting-lab)
