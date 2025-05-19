# Financial Portfolio Optimization Tool

This project is a **university assignment** demonstrating portfolio optimization using evolutionary algorithms. The goal is to allocate capital across various assets (stocks, ETFs, bonds, commodities) to maximize the risk-adjusted return (Sharpe ratio) while adhering to predefined constraints.

---

## Features
- **Asset Loading**: Assets are loaded from `assets.csv`, which includes:
  - Asset name
  - Type (`etf`, `stock`, `bond`, `commodity`)
  - Historical return (%)
  - Risk (volatility %)
  - Beta (market sensitivity)
- **Constraints**:
  - Minimum 15% allocation to bonds
  - Minimum 5% allocation to commodities
  - Each asset must have at least 2% allocation if included
  - Maximum 30% allocation to a single asset
  - Minimum of 5 distinct assets in the portfolio
- **Metrics**:
  - **Sharpe ratio**: (Portfolio return - Risk-free rate) / Portfolio risk
  - **Treynor ratio**: (Portfolio return - Risk-free rate) / Portfolio beta

---

## Algorithm
The optimization uses **Differential Evolution (DE)**, a stochastic population-based algorithm inspired by biological evolution. Steps include:
1. **Population Initialization**: Generate random candidate portfolios.
2. **Mutation & Crossover**: Combine weights from different candidates to explore new solutions.
3. **Selection**: Retain portfolios with the highest Sharpe ratio.
4. **Termination**: Stop after reaching maximum iterations (`maxiter=1000`) or convergence.

---

## Optimization Process
1. **Fitness Function**: Evaluates portfolios based on:
   - Sharpe ratio (to maximize).
   - Penalties for constraint violations (e.g., insufficient bond allocation, over-concentration in a single asset).
2. **Diversification**: Favors combinations of assets with low correlation to reduce overall risk.
3. **Balance**: Balances high-return/high-risk assets (stocks) with stable assets (bonds).
