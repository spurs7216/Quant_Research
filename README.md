# Quantitative Portfolio optimization using Crypto data

by Joseph Shih, May 26 2025

## **Data Overview**
This repository contains tick-level perpetual futures data for ten major cryptocurrencies, sampled at 1-hour intervals over the period January 1, 2022 through November 30, 2023.
*  Assets covered (symbols):BTCUSDT,ETHUSDT, LINKUSDT, XRPUSDT, SOLUSDT, BNBUSDT,MATICUSDT, GALAUSDT, GMTUSDT, ADAUSDT
*  Time span: 2022-01-01 00:00:00 UTC through 2023-11-30 23:00:00 UTC
*  Frequency: 1-hour bars (744 bars per 31-day month; 720–744 per month depending on month length)
*  Source: Binance 
*  Features:
    *  open_time:	Interval start timestamp in milliseconds since Unix epoch
    *  high:	Highest traded price during the interval
    *  low:	Lowest traded price during the interval
    *  close:	Closing price at the end of the interval
    *  volume:	Total traded base‐asset volume in the interval
    *  close_time:	Interval end timestamp in milliseconds since Unix epoch
    *  quote_volume:	Total traded quote‐asset volume (i.e., price × volume)
    *  count:	Number of individual trades executed during the interval
    *  taker_buy_volume:	Base‐asset volume purchased by taker orders (market buys)
    *  taker_buy_quote_volume:	Quote‐asset volume purchased by taker orders

## 📊Simple
Perform a deep dive into crypto return dynamics and test a simple mean-reversion strategy.
1. **Momentum & Volatility Factors**  
   - **Momentum (L = 24 h)**: cumulative return over the past 24 hours  
   - **Volatility (L = 24 h)**: rolling standard deviation of hourly returns  
   - Tag top/bottom coins each period and evaluate next‐hour return  

2. **Autoregressive Modeling**  
   - Fit **AR(p)** models to each coin’s return series  
   - Inspect coefficients and residuals for serial correlation  

3. **Stationarity & Mean-Reversion Tests**  
   - **Augmented Dickey–Fuller (ADF)** to confirm stationarity of returns  
   - Estimate **half-life** of mean reversion via AR(1) decay  
   - Demonstrate stronger mean-reverting behavior in certain coins  

4. **Cointegration Analysis**  
   - **Cointegration tests** between coin pairs/triples  
     - **CADF** (Christiano–Dickey–Fuller)  
     - **Johansen’s test** for multivariate cointegration  
   - Identify candidate pairs that share long-run equilibrium  

5. **Pairs-Trading / Mean-Reversion Strategy**  
   - Construct **spread** \(Sₜ = price₁ₜ – β·price₂ₜ\) using OLS‐estimated β  
   - Backtest a simple rule:  
     - Go **long** spread when it is sufficiently below its rolling mean  
     - Go **short** when above its rolling mean  
   - Evaluate P&L, Sharpe ratio, and turnover  

## 📊Machine learning strategy
Experiment with a range of supervised and unsupervised learning methods — plus a reinforcement‐learning agent — to generate trading signals and build a crypto portfolio.
1. **Supervised Classification Models**  
   - **Logistic Regression (L1)** with cross‐validated regularization strength.  
   - **Random Forest** for capturing nonlinear interactions among features.  
   - **LassoCV** as a built‐in feature‐selecting linear model.  
   - **XGBoost & CatBoost**: gradient‐boosted tree ensembles, grid‐searched over depth, learning rate, iterations.  
   - **Signal construction**:  
     - Classifier probability > 0.5 → go **long**, < 0.5 → go **short**.  
     - Compute **Sharpe ratio** on validation/test using `compute_sharpe_from_signal()`.  
   - **Model selection** picks hyperparameters that maximize out‐of‐sample Sharpe.

2. **Unsupervised Feature Learning (Autoencoders)**  
   - **Seq-to-Seq Autoencoder**: compress rolling‐window returns into a low-dimensional latent space.  
   - **Denoising Autoencoder**: train to reconstruct original returns from noisy inputs, yielding robust embeddings.  
   - **Use case**: append encoded features to the supervised models to improve generalization.

3. **Reinforcement Learning Agent**  
   - **Environment**: `CryptoPortfolioEnv` simulates a small portfolio with transaction costs and slippage.  
   - **Algorithm**: Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) for stable policy updates.  
   - **Workflow**:  
     1. Collect rollouts in training environment  
     2. Update policy/value networks  
     3. Backtest final policy on held‐out data  

## 📊Hidden Markov Model
Leverage HMMs to identify latent market regimes and drive a simple regime‐aware portfolio.
1. **Training the HMM**  
   - Fit a **Gaussian HMM** with \(K\) hidden states (e.g. \(K=2\) or \(3\)) to the multivariate return+feature series.  
   - Use the **Baum–Welch** algorithm to estimate transition probabilities and per-state emission parameters.  
   - Inspect the fitted means and variances in each state to interpret regimes (e.g. “low-vol mean-reversion” vs. “high-vol trending”).

2. **Regime Decoding & Probabilities**  
   - Apply the **Viterbi algorithm** to assign the most likely state sequence to the historical data.  
   - Compute **smoothed state-posterior probabilities** for each minute, giving a soft view of regime membership.

3. **Regime‐Based Trading Rules**  
   - Define a **long/short rule** for each regime, for example:  
     - **Regime 0 (Mean‐reverting):** Go **long** when posterior > 0.8, flat otherwise.  
     - **Regime 1 (Trending):** Use a simple momentum signal (e.g. 1‐hour moving average crossover).  
   - Optionally, tailor **position sizing** by regime volatility (e.g. smaller positions in high-vol states).
