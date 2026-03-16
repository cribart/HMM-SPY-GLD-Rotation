# 📈 HMM Algorithmic Portfolio Manager (SPY / GLD)

An algorithmic trading and portfolio management framework built in Python. This model uses **Hidden Markov Models (GaussianHMM)** to detect market regimes and dynamically rotate capital between equities (SPY) and defensive assets (GLD) to minimize drawdowns during market crashes.

## 🧠 Strategy Logic
The market is not static; it transitions through different probabilistic "states" (e.g., bull market, high-volatility crash, sideways chop). 
* **🟢 Risk-On (Bull State):** When the HMM identifies a low-volatility, positive momentum environment, the portfolio allocates 100% to the S&P 500 (`SPY`).
* **🔴 Risk-Off (Crash State):** When the model detects a regime shift characterized by spiking volatility and negative expected returns, it liquidates equities and rotates 100% into Gold (`GLD`) to protect capital.
* **🟡 Partial Defense:** If the HMM is clear but momentum is negative, the model dynamically shifts to a 40/60 SPY/GLD allocation.

## ⚙️ Key Features
* **Machine Learning Regime Detection:** Utilizes `hmmlearn` to analyze log returns, rolling volatility, and VIX data.
* **Walk-Forward Backtesting:** Avoids overfitting by continuously retraining the HMM on a rolling window.
* **Advanced Statistical Validation:** Implements Marcos Lopez de Prado's **Probabilistic Sharpe Ratio (PSR)** and **Minimum Track Record Length (MinTRL)**.
* **Log-Likelihood Optimization:** Dynamically selects the optimal lookback windows based on maximum likelihood estimation.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `hmmlearn`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `yfinance`, `matplotlib`
