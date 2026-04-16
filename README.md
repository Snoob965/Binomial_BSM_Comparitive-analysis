# 📉 Options Pricing: CRR Binomial Tree vs Black-Scholes

A quantitative analysis engine built in Python to evaluate and compare the continuous-time **Black-Scholes-Merton (BSM)** model against the discrete-time **Cox-Ross-Rubinstein (CRR) Binomial Tree** model. 

This project simulates Out-of-The-Money (OTM) SPY ETF options and benchmarks the computational latency, pricing accuracy, and parameter sensitivities of both models.

---

## 🧠 The Quantitative Theory

### The Black-Scholes Model (Continuous Time)
The BSM model provides a closed-form algebraic solution for European options. It assumes stock prices follow a geometric Brownian motion with constant drift and volatility. Because it is a closed-form formula, it calculates in fractions of a millisecond. However, it cannot easily handle American options (early exercise) or discrete dividend payouts.

### The CRR Binomial Tree (Discrete Time)
The Binomial Tree is a lattice-based model that slices time to maturity ($T$) into $N$ discrete steps. At each node, the asset price can either move "up" or "down" based on mathematically calibrated probabilities derived from the risk-free rate and volatility. 
* **The Advantage:** It provides extreme flexibility. It can easily be modified to price American options by checking the intrinsic value at every single node during backward induction.
* **The Cost:** It is computationally expensive. As $N$ approaches infinity, the Binomial Tree perfectly converges to the Black-Scholes price, but the CPU latency grows exponentially. 

---

## 🏎️ Computational Architecture & Benchmarking

This engine is built to mirror desk-level analysis tools. Instead of relying on slow Python loops for the tree generation, the asset paths are vectorized using `numpy` arrays.

**Performance Benchmark ($N=500$ Steps):**
* The continuous BSM formula executes in **~0.05 ms**.
* The discrete $N=500$ Binomial Tree executes in **~1.50 ms**.
* While the tree is computationally heavier, $N=500$ achieves a pricing error margin of less than $\$0.005$ compared to the BSM limit.

---

## 📊 Interactive 6-Panel Dashboard

The engine utilizes `PyQt5` to bypass standard Tkinter limitations, generating a standalone, interactive 3D dashboard. It simultaneously calculates and plots:
1. **Pricing Convergence:** The discrete tree approaching the continuous BSM limit.
2. **Absolute Error Decay:** The diminishing pricing error as $N \rightarrow \infty$.
3. **Vega Profiling:** Option sensitivity to Implied Volatility ($\sigma$).
4. **Theta Profiling:** Option decay against Time to Maturity ($T$).
5. **Rho Profiling:** Sensitivity to Risk-Free Interest Rates ($r$).
6. **Strike Sensitivity:** Price variation against Strike Price ($K$).

> 

https://github.com/user-attachments/assets/3251de04-cd0c-4bed-b9c5-9f5d16a84e58



---

## 💻 Quick Start

```bash
# Clone the repository
git clone [https://github.com/Snoob965/Binomial_BSM_Comparitive-analysis.git](https://github.com/Snoob965/Binomial_BSM_Comparitive-analysis.git)
cd Binomial_BSM_Comparitive-analysis

# Install dependencies
pip install numpy scipy matplotlib PyQt5

# Run the engine and launch the master dashboard
python analyzer.py
