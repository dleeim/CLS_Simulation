-----

# Crypto-Linked Security (CLS) Pricing and Risk Analysis Engine

This project demonstrates a basic quantitative finance workflow for pricing of a path-dependent derivative Crypto-Linked Security (CLS) based on BTC/USDT in Binance options market data.

-----

## Key Features

  * **Implied Volatility Surface Construction:** Ingests raw options data to calculate implied volatilities, then uses 2D spline interpolation to construct a smooth, continuous IV surface that captures both the **Volatility Smile/Skew** and the **Term Structure**.
  * **Dynamic Monte Carlo Engine:** Prices the CLS using a Monte Carlo simulation that incorporates a **dynamic, non-constant volatility**. The engine uses the constructed IV Term Structure to look up the appropriate volatility at each time step, providing a far more realistic price path generation.
  * **Comprehensive Risk Analysis (The Greeks):** Calculates the greeks for the CLS product:

-----

## Workflow

The project follows a systematic, real-world quantitative analysis workflow:

1.  **Data Ingestion & Cleaning (`src/` & `Cleaning&EDA&FE_ENG.ipynb`):**

      * Raw BTC spot and options data **`EOH summary`** are programmatically downloaded from Binance.
      * Data is cleaned, pre-processed, and engineered into a usable format for analysis.
      * Implied Volatility is calculated for each option using the Black-Scholes model and a root-finding algorithm **`brentq`**.
      * Compared calculated IV with mark IV provided in Binance and achieved 99.9% of data having +-5% error.
      ![iv_error_distribution](outputs/iv_error_distribution.jpg)

2.  **Volatility Surface Modeling (`src/iv_engine.py` & `iv_surface_analysis.ipynb`):**

      * A continuous **Implied Volatility Surface** is fitted to the market data using **`Stochastic Volatility Inspired`** model and a spline interpolation model **`RectBivariateSpline`**. This serves as the primary input for all subsequent models.
      * The **IV Skews** and **ATM Volatility Term Structure** are extracted from the surface to be used in the simulation.

3.  **CLS Product Design & Pricing (`CSL_analysis.ipynb`):**

      * A custom CLS product with features like Autocall (early redemption) and a Knock-In (KI) barrier is designed.
      * The product is priced using a vectorized **Monte Carlo simulation engine** that dynamically references the volatility term structure at each time step.
      * Used finite difference method to calculate greeks of CLS.