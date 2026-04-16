import numpy as np
import scipy.stats as si
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time

class OptionsPricingEngine:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.type = option_type.lower()

    def black_scholes_price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.type == 'call':
            return self.S * si.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2) - self.S * si.norm.cdf(-d1)

    def black_scholes_greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = si.norm.cdf(d1) if self.type == 'call' else si.norm.cdf(d1) - 1
        gamma = si.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * si.norm.pdf(d1) * np.sqrt(self.T)
        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega}

    def binomial_tree_price_and_greeks(self, N):
        if self.T <= 0:
            return max(self.S - self.K, 0) if self.type == 'call' else max(self.K - self.S, 0), 0.0
            
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        ST = np.array([self.S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
        
        if self.type == 'call':
            C = np.maximum(ST - self.K, 0)
        else:
            C = np.maximum(self.K - ST, 0)

        for i in range(N - 1, 0, -1):
            C = np.exp(-self.r * dt) * (p * C[1:i+2] + (1 - p) * C[0:i+1])
            
        C_up, C_down = C[1], C[0]
        S_up, S_down = self.S * u, self.S * d
        discrete_delta = (C_up - C_down) / (S_up - S_down) if (S_up - S_down) != 0 else 0
        
        price = np.exp(-self.r * dt) * (p * C_up + (1 - p) * C_down)
        return price, discrete_delta

    def benchmark_performance(self, N=1000):
        print(f"\n--- Performance Benchmarking (N={N} Steps) ---")
        start_bs = time.perf_counter()
        bs_price = self.black_scholes_price()
        bs_time = time.perf_counter() - start_bs
        
        start_bt = time.perf_counter()
        bt_price, _ = self.binomial_tree_price_and_greeks(N)
        bt_time = time.perf_counter() - start_bt
        
        print(f"Black-Scholes:  {bs_time*1000:.4f} ms | Price: ${bs_price:.4f}")
        print(f"Binomial Tree:  {bt_time*1000:.4f} ms | Price: ${bt_price:.4f}")
        print(f"Latency Ratio:  Tree is {bt_time/bs_time:.1f}x slower than BSM")

    def plot_master_dashboard(self, max_steps=150):
        print("\nCalculating 6-panel dashboard data... Please wait ~5-10 seconds.")
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Master Options Pricing Dashboard: Binomial Tree vs Black-Scholes', fontsize=16, fontweight='bold')

        bs_base_price = self.black_scholes_price()
        steps = np.arange(10, max_steps, 5)
        bt_prices = [self.binomial_tree_price_and_greeks(n)[0] for n in steps]
        errors = [abs(p - bs_base_price) for p in bt_prices]

        axs[0, 0].plot(steps, bt_prices, label='CRR Binomial Tree', color='#1f77b4', marker='.')
        axs[0, 0].axhline(y=bs_base_price, color='red', linestyle='--', label=f'BSM Limit (${bs_base_price:.4f})')
        axs[0, 0].set_title('1. Pricing Convergence')
        axs[0, 0].set_ylabel('Option Price ($)')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        axs[0, 1].plot(steps, errors, color='purple', marker='.')
        axs[0, 1].set_title('2. Absolute Error Decay')
        axs[0, 1].set_ylabel('Absolute Error ($)')
        axs[0, 1].grid(True, alpha=0.3)

        vols = np.linspace(0.05, 0.60, 15)
        bs_vols, bt_vols = [], []
        orig_sigma = self.sigma
        for v in vols:
            self.sigma = v
            bs_vols.append(self.black_scholes_price())
            bt_vols.append(self.binomial_tree_price_and_greeks(100)[0])
        self.sigma = orig_sigma

        axs[1, 0].plot(vols, bs_vols, label='Black-Scholes', color='red', linewidth=2)
        axs[1, 0].plot(vols, bt_vols, label='Binomial Tree (N=100)', color='blue', linestyle='--')
        axs[1, 0].set_title('3. Sensitivity to Volatility (Vega)')
        axs[1, 0].set_xlabel('Implied Volatility ($\sigma$)')
        axs[1, 0].set_ylabel('Option Price ($)')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        maturities = np.linspace(0.05, 2.0, 15)
        bs_mats, bt_mats = [], []
        orig_T = self.T
        for t_val in maturities:
            self.T = t_val
            bs_mats.append(self.black_scholes_price())
            bt_mats.append(self.binomial_tree_price_and_greeks(100)[0])
        self.T = orig_T

        axs[1, 1].plot(maturities, bs_mats, color='red', linewidth=2)
        axs[1, 1].plot(maturities, bt_mats, color='blue', linestyle='--')
        axs[1, 1].set_title('4. Sensitivity to Time to Maturity (Theta)')
        axs[1, 1].set_xlabel('Time to Maturity ($T$ in years)')
        axs[1, 1].grid(True, alpha=0.3)

        rates = np.linspace(0.0, 0.10, 15)
        bs_rates, bt_rates = [], []
        orig_r = self.r
        for r_val in rates:
            self.r = r_val
            bs_rates.append(self.black_scholes_price())
            bt_rates.append(self.binomial_tree_price_and_greeks(100)[0])
        self.r = orig_r

        axs[2, 0].plot(rates, bs_rates, color='red', linewidth=2)
        axs[2, 0].plot(rates, bt_rates, color='blue', linestyle='--')
        axs[2, 0].set_title('5. Sensitivity to Risk-Free Rate (Rho)')
        axs[2, 0].set_xlabel('Risk-Free Rate ($r$)')
        axs[2, 0].set_ylabel('Option Price ($)')
        axs[2, 0].grid(True, alpha=0.3)

        strikes = np.linspace(480, 540, 15)
        bs_strikes, bt_strikes = [], []
        orig_K = self.K
        for k_val in strikes:
            self.K = k_val
            bs_strikes.append(self.black_scholes_price())
            bt_strikes.append(self.binomial_tree_price_and_greeks(100)[0])
        self.K = orig_K

        axs[2, 1].plot(strikes, bs_strikes, color='red', linewidth=2)
        axs[2, 1].plot(strikes, bt_strikes, color='blue', linestyle='--')
        axs[2, 1].set_title('6. Sensitivity to Strike Price')
        axs[2, 1].set_xlabel('Strike Price ($K$)')
        axs[2, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)
        plt.savefig("master_dashboard.png", dpi=300, bbox_inches='tight')
        print("Success: Saved 'master_dashboard.png'.")
        print("Opening interactive dashboard...")
        plt.show()

if __name__ == "__main__":
    spy_options = OptionsPricingEngine(S=510, K=515, T=0.25, r=0.045, sigma=0.12, option_type='call')
    print("--- SPY Options Greek Calculation ---")
    greeks = spy_options.black_scholes_greeks()
    _, bt_delta = spy_options.binomial_tree_price_and_greeks(N=500)
    print(f"BSM Delta:  {greeks['Delta']:.4f}\nTree Delta: {bt_delta:.4f}\nBSM Gamma:  {greeks['Gamma']:.4f}\nBSM Vega:   {greeks['Vega']:.4f}")
    spy_options.benchmark_performance(N=500)
    spy_options.plot_master_dashboard(max_steps=150)