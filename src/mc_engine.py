import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def generate_stock_paths(S0, T, r, N_STEPS, N_SIMS, term_structure_func) -> np.ndarray:
    
    dt = T / N_STEPS
    paths = np.zeros((N_STEPS + 1, N_SIMS))
    paths[0] = S0
    
    # Generate random numbers for all paths at once
    Z = np.random.standard_normal((N_STEPS, N_SIMS))
    
    for t in range(1, N_STEPS + 1):
        # Calculate remaining time to maturity
        remaining_time = T - (t - 1) * dt
        
        # Look up volatility from term structure
        sigma_t = term_structure_func(remaining_time)
        
        # Generate next price step
        paths[t, :] = paths[t-1, :] * np.exp(
            (r - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * Z[t-1, :]
        )
    
    return paths


def evaluate_cls_payoffs(paths: np.ndarray, S0, T, N_STEPS, 
                         autocall_time, autocall_level, 
                         ki_barrier_level, coupon_rate):

    dt = T / N_STEPS
    N_SIMS = paths.shape[1]
    
    # Initialize payoffs
    payoffs = np.zeros(N_SIMS)
    
    # 1. Autocall Check
    autocall_step = int(autocall_time / dt)
    autocall_price_level = autocall_level * S0
    price_at_autocall = paths[autocall_step, :]
    
    autocalled_mask = price_at_autocall >= autocall_price_level
    autocall_payoff = 1 + (coupon_rate * autocall_time)  # Principal + prorated coupon
    payoffs[autocalled_mask] = autocall_payoff
    
    # 2. Final Payoff Check (for non-autocalled paths)
    not_autocalled_mask = ~autocalled_mask
    
    if np.any(not_autocalled_mask):
        # Check knock-in barrier breach
        ki_barrier_price = ki_barrier_level * S0
        min_price_path = np.min(paths[:, not_autocalled_mask], axis=0)
        ki_triggered_mask = min_price_path < ki_barrier_price
        
        # Final payoffs for non-autocalled paths
        final_prices = paths[-1, not_autocalled_mask]
        
        # KI NOT triggered: Principal + full coupon
        final_payoff_no_ki = 1 + (coupon_rate * T)
        
        # KI triggered: Principal * (Final Price / S0)
        final_payoff_ki = final_prices / S0
        
        # Assign final payoffs
        not_autocalled_indices = np.where(not_autocalled_mask)[0]
        payoffs[not_autocalled_indices[~ki_triggered_mask]] = final_payoff_no_ki
        payoffs[not_autocalled_indices[ki_triggered_mask]] = final_payoff_ki[ki_triggered_mask]
    
    # Calculate statistics
    stats = {
        'autocall_rate': np.mean(autocalled_mask),
        'ki_trigger_rate': np.mean(min_price_path < ki_barrier_price) if np.any(not_autocalled_mask) else 0,
        'mean_payoff': np.mean(payoffs),
        'payoff_std': np.std(payoffs),
        'min_payoff': np.min(payoffs),
        'max_payoff': np.max(payoffs)
    }
    
    return payoffs, stats

def evaluate_cls_greeks(S0, T, r, N_STEPS, N_SIMS, term_structure_func, autocall_time, autocall_level, ki_barrier_level, coupon_rate):

    # Parameters
    k = 0.005
    eps = S0*k
    eps_iv = 0.01
    S0_up = S0*(1+k)
    S0_down = S0*(1-k)

    # paths and payoffs
    paths = generate_stock_paths(S0, T, r, N_STEPS, N_SIMS, term_structure_func)
    payoffs, stats = evaluate_cls_payoffs(paths, S0, T, N_STEPS, autocall_time, autocall_level, ki_barrier_level, coupon_rate)
    paths_up = generate_stock_paths(S0_up, T, r, N_STEPS, N_SIMS, term_structure_func)
    payoffs_up, stats_up = evaluate_cls_payoffs(paths_up, S0_up, T, N_STEPS, autocall_time, autocall_level, ki_barrier_level, coupon_rate)
    paths_up = generate_stock_paths(S0_up, T, r, N_STEPS, N_SIMS, term_structure_func)
    payoffs_up, stats_up = evaluate_cls_payoffs(paths_up, S0_up, T, N_STEPS, autocall_time, autocall_level, ki_barrier_level, coupon_rate)
    paths_down = generate_stock_paths(S0_down, T, r, N_STEPS, N_SIMS, term_structure_func)
    payoffs_down, stats_down = evaluate_cls_payoffs(paths_down, S0_down, T, N_STEPS, autocall_time, autocall_level, ki_barrier_level, coupon_rate)

    term_structure_func_up = lambda t: term_structure_func(t) + eps_iv
    term_structure_func_down = lambda t: term_structure_func(t) - eps_iv
    paths_up_iv = generate_stock_paths(S0, T, r, N_STEPS, N_SIMS, term_structure_func_up)
    payoffs_up_iv, stats_up_iv = evaluate_cls_payoffs(paths_up_iv, S0, T, N_STEPS, autocall_time, autocall_level, ki_barrier_level, coupon_rate)
    paths_down_iv = generate_stock_paths(S0, T, r, N_STEPS, N_SIMS, term_structure_func_down)
    payoffs_down_iv, stats_down_iv = evaluate_cls_payoffs(paths_down_iv, S0, T, N_STEPS, autocall_time, autocall_level, ki_barrier_level, coupon_rate)

    # Greeks
    return {
        'delta' : S0*(stats_up['mean_payoff'] - stats_down['mean_payoff'])/(2*eps),
        'gamma' : S0*(stats_up['mean_payoff'] - stats['mean_payoff'] + stats_down['mean_payoff'])/(eps**2),
        'vega'  : S0*(stats_up_iv['mean_payoff'] - stats_down_iv['mean_payoff'])/(2*eps)
    }


def plot_simulation_paths(paths: np.ndarray, S0, T, N_STEPS,
                         autocall_time, autocall_level,
                         ki_barrier_level, n_paths_display):

    time_grid = np.linspace(0, T, N_STEPS + 1)
    
    # Select subset of paths to display
    n_display = min(n_paths_display, paths.shape[1])
    display_indices = np.random.choice(paths.shape[1], n_display, replace=False, )
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    for i in display_indices:
        ax.plot(time_grid, paths[:, i], alpha=0.3, color='blue', linewidth=0.5)
    
    ax.axhline(y=S0, color='black', linestyle='-', linewidth=2, label=f'Initial Price (${S0:.0f})')
    ax.axhline(y=autocall_level * S0, color='green', linestyle='--', linewidth=2, 
               label=f'Autocall Level (${autocall_level * S0:.0f})')
    ax.axhline(y=ki_barrier_level * S0, color='red', linestyle='--', linewidth=2,
               label=f'KI Barrier (${ki_barrier_level * S0:.0f})')
    
    ax.axvline(x=autocall_time, color='orange', linestyle=':', linewidth=2,
               label=f'Autocall Time ({autocall_time*12:.0f} months)')
    
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('BTC/USDT')
    ax.set_title(f'CLS Monte Carlo Simulation\n{n_display} Sample Paths out of {paths.shape[1]:,} Total')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    y_min = max(0, np.min(paths) * 0.95)
    y_max = np.max(paths) * 1.05
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.show()

def plot_payoff_distribution(payoffs: np.ndarray):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
    
    # Histogram
    ax1.hist(payoffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(payoffs), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(payoffs):.3f}')
    ax1.set_xlabel('Payoff')
    ax1.set_ylabel('Frequency')
    ax1.set_title('CLS Payoff Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(payoffs, vert=True)
    ax2.set_ylabel('Payoff')
    ax2.set_title('CLS Payoff Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()