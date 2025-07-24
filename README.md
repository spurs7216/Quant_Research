# Quantitative Portfolio Optimization using Crypto Data

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

## 📊Pairs
Perform a research on crypto pairs trading and test a simple mean-reversion strategy.
1. **Stationarity & Mean-Reversion Tests**  
   - **Augmented Dickey–Fuller (ADF)** to confirm stationarity of returns  
   - Estimate **half-life** of mean reversion via AR(1) decay  
   - Demonstrate stronger mean-reverting behavior in certain coins  

2. **Cointegration Analysis**  
   - **Cointegration tests** between coin pairs/triples  
     - **CADF** (Christiano–Dickey–Fuller)  
     - **Johansen’s test** for multivariate cointegration  
   - Identify candidate pairs that share long-run equilibrium  

3. **Pairs-Trading / Mean-Reversion Strategy**  
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

## 📊Alpha_in_crypto
Implement a diverse set of systematic strategies to uncover short‐term alphas across multiple coins, based on both machine‐learning and classic technical indicators.
1. **Rolling Ridge Regression**
   - Setup: Use a rolling window to form training set.
   - Features: For each asset and look‐back date, stack lagged returns and technical covariates, rolling volatility change, imbalance, and distance to extremes.
   - Model: Fit a scikit‐learn Ridge regression to predict next‐period return​.
   - Positioning: Cross‐sectionally threshold predicted alphas at the 95% quantile and scale exposures by a gross leverage factor (0.5) adjusted for each asset’s volatility. 

2. **Moving‐Average Crossover**
   - Short & Long MAs: Compute a 12‐period and 24‐period simple moving average of closing prices.
   - Signal Construction: Take the difference Δ=SMA12−SMA24​; only trade when ∣Δ∣ exceeds the cross‐sectional 95th95th percentile.
   - Weighting: Assign long/short weights proportional to sign(Δ) and scale by inverse ATR24 to control for volatility.

3. **Combined Indicator Signal**
   - Indicator Suite:
      - RSI12 (overbought/oversold)
      - Stochastic %K (12)
      - Ease of Movement (EMV)
      - TRIX (12)
      - ADX12 with +DI/–DI directional filters
   - Voting Mechanism: At each minute and for each coin, count how many indicators signal bullish vs. bearish.
   - Trading Rule: Go long on coins where bullish votes exceed bearish votes, short if the reverse, and flat otherwise.

4. **Soft‐Voting with Threshold & Volume Filter**
   - Soft Score: Compute a weighted sum of individual indicator signals, where each indicator’s weight combines its historical accuracy and a regime‐specific momentum or reversal factor.
   - Thresholding: Zero out scores with absolute value below θ=0.3.
   - Volume Filter: Exclude low‐liquidity coins by applying a minimum volume‐based screen before executing trades.

5. **Performance Evaluation**
   - Metrics Calculation:
      - AvgRet: Mean hourly return
      - Sharpe: Annualized ratio (√8760 hours)
      - MaxDraw: Maximum drawdown
      - AvgWin/AvgLoss: Mean positive/negative returns
   - Summary & Plotting: Aggregate results into a DataFrame and visualize cumulative P&L across all four methods to compare risk‐adjusted performance.