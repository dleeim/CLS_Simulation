import pandas as pd, numpy as np, matplotlib.pyplot as plt  
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline

def svi(k, a, b, s, rho, m):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + s**2))

def fit_svi_slice(k, iv, T, x0=None):
    total_var_mkt = (iv ** 2) * T          # convert IV -> total variance

    def error(p):
        return svi(k, *p) - total_var_mkt

    if x0 is None:
        x0 = np.zeros(5)                   # [a,b,s,rho,m]
        x0[2] = 0.1                        # small positive s

    res = least_squares(
        error,
        x0=x0,
        bounds=([-np.inf, 0, 1e-9, -0.999, -np.inf],  # b>0, s>0, |rho|<1
                [ np.inf, np.inf, np.inf, 0.999, np.inf]),
        max_nfev=10_000,
        xtol=1e-8,
        ftol=1e-8,
    )
    return res.x

def build_iv_surface_spline(df: pd.DataFrame):
    
    # define strike & tenor grids present in the snapshot
    k_grid   = np.unique(df["log_moneyness"].values.astype(float))
    T_unique = np.unique(df["time_to_expiry"].values.astype(float))
    Z_raw = np.empty((k_grid.size, T_unique.size))

    # fit raw‑SVI slice per maturity
    last_params = np.array([0.0, 0.1, 0.1, 0.0, 0.0])   # seed for first slice
    for j, T in enumerate(T_unique):
        rows   = df["time_to_expiry"] == T
        k_vec  = df.loc[rows, "log_moneyness"].to_numpy(float)
        iv_vec = df.loc[rows, "iv_calc"].to_numpy(float)

        params      = fit_svi_slice(k_vec, iv_vec, T, x0=last_params)
        last_params = params
        Z_raw[:, j] = np.sqrt((svi(k_grid, *params) / T).clip(0.001,np.inf))

    # build bivariate spline in (k, T)
    return RectBivariateSpline(k_grid, T_unique, Z_raw, kx=3, ky=2)

def plot_iv_surface(K,T,Z):
    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection="3d")

    K_mesh, T_mesh = np.meshgrid(K, T, indexing="ij")

    surf = ax.plot_surface(
        K_mesh, T_mesh, Z,
        rstride=1, cstride=1, linewidth=0, antialiased=False,
        cmap="viridis", alpha=0.9
    )

    ax.set_xlabel("log-moneyness  k = ln(K/S)")
    ax.set_ylabel("Time to expiry  T  (years)")
    ax.set_zlabel(r"Implied volatility $\sigma$")
    ax.set_title("SVI-fitted IV surface")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()

def plot_iv_skew(K,T,Z):
    plt.figure(figsize=(8, 5))
    for idx in range(len(T)):            
        plt.plot(K, Z[:, idx], label=f"T ≈ {T[idx]*365:.0f} d")

    plt.legend()
    plt.xlabel("log-moneyness k")
    plt.ylabel("Implied volatility")
    plt.title("Cross-sectional IV skews at selected tenors")
    plt.grid(alpha=0.3)
    plt.show()

def plot_term_structure(K,T,Z):
    plt.figure(figsize=(8, 5))
    for idx in range(len(K)):
        plt.plot(T, Z[idx, :], label=f"K ≈ {K[idx]:.0f}")

    plt.legend()
    plt.xlabel("Time to Expiry (years)")
    plt.ylabel("Implied volatility")
    plt.title(f"Cross-sectional IV term structure at log moneyness = {K[idx]}")
    plt.grid(alpha=0.3)
    plt.show()