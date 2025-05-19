import numpy as np
import csv
from scipy.optimize import differential_evolution

# ZÁKLADNÉ NASTAVENIA
total_investment = 10000
min_bonds_allocation = 0.15
min_commodities_allocation = 0.05
min_asset_allocation = 0.02
max_single_asset_allocation = 0.30
required_sharpe_ratio = 0.6
risk_free_rate = 0.015
min_number_of_assets = 5

# NAČÍTANIE AKTÍV Z CSV
etfs = []
stocks = []
bonds = []
commodities = []
betas = []

with open('assets.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        asset = (
            row['name'],
            float(row['return']),
            float(row['risk']),
            float(row['beta'])
        )
        if row['type'] == 'etf':
            etfs.append(asset)
        elif row['type'] == 'stock':
            stocks.append(asset)
        elif row['type'] == 'bond':
            bonds.append(asset)
        elif row['type'] == 'commodity':
            commodities.append(asset)
        betas.append(float(row['beta']))

assets = etfs + stocks + bonds + commodities
num_assets = len(assets)
returns = np.array([a[1] for a in assets])
risks = np.array([a[2] for a in assets])
betas = np.array(betas)


# FITNESS FUNKCIA
def fitness(weights):
    weights = np.array(weights)
    weights /= np.sum(weights)

    bond_indices = slice(len(etfs) + len(stocks), len(etfs) + len(stocks) + len(bonds))
    bond_sum = np.sum(weights[bond_indices])
    commodity_sum = np.sum(weights[-len(commodities):])
    max_single_weight = np.max(weights)

    penalty = 0

    if bond_sum < min_bonds_allocation:
        penalty += 1e5 * (min_bonds_allocation - bond_sum)  # Zvýšená penalizácia
    if commodity_sum < min_commodities_allocation:
        penalty += 1e5 * (min_commodities_allocation - commodity_sum)

    violating_assets = np.where((weights > 0.001) & (weights < min_asset_allocation))[0]
    if len(violating_assets) > 0:
        penalty += 1e5 * len(violating_assets)

    if max_single_weight > max_single_asset_allocation:
        penalty += 1e5 * (max_single_weight - max_single_asset_allocation)

    num_assets_used = np.sum(weights >= min_asset_allocation)
    if num_assets_used < min_number_of_assets:
        penalty += 1e5 * (min_number_of_assets - num_assets_used)

    port_return = np.dot(weights, returns)
    port_risk = np.dot(weights, risks)
    sharpe_ratio = (port_return - risk_free_rate) / port_risk

    return -sharpe_ratio + penalty


# OPTIMALIZÁCIA
bounds = [(0, 1)] * num_assets
result = differential_evolution(
    fitness, bounds, strategy='best1bin',
    maxiter=1000, popsize=40, tol=1e-6)

optimal_weights = result.x / np.sum(result.x)
final_return = np.dot(optimal_weights, returns)
final_risk = np.dot(optimal_weights, risks)
final_sharpe = (final_return - risk_free_rate) / final_risk

portfolio_beta = np.dot(optimal_weights, betas)
final_treynor = (final_return - risk_free_rate) / portfolio_beta

# VÝSTUP
print("=== Optimálne portfólio ===\n")
print(f"Celková investícia: {total_investment:.2f} EUR")
print(f"Min. podiel do dlhopisov: {min_bonds_allocation * 100:.1f}%")
print(f"Min. podiel do komodít: {min_commodities_allocation * 100:.1f}%")
print(f"Min. alokácia pre použité aktívum: {min_asset_allocation * 100:.1f}%")
print(f"Max. alokácia na jedno aktívum: {max_single_asset_allocation * 100:.1f}%")
print(f"Bezriziková úroková miera: {risk_free_rate * 100:.2f}%\n")

print("Zloženie portfólia:\n")
for i, weight in enumerate(optimal_weights):
    if weight >= 0.001:
        name = assets[i][0]
        amount = weight * total_investment
        print(f"{name:40s} - {weight * 100:6.2f}%   = {amount:8.2f} EUR")

print(f"\nOčakávaný výnos: {final_return * 100:.2f}% ročne")
print(f"Odhadované riziko (volatilita): {final_risk * 100:.2f}%")
print(f"Sharpe ratio: {final_sharpe:.2f}")
print(f"Treynor ratio: {final_treynor:.2f}")