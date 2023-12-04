import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import os

def get_column_data(df, column_name):
    if column_name in df.columns:
        return df[column_name]
    else:
        return df[df.columns[1]]

grid_prices = pd.read_csv('prices_2019_hours.txt', sep=',', parse_dates=['DATE'])
temps = pd.read_csv('temps_2019_hours.txt', sep=',', parse_dates=['DATE'])

k = 0.1 
T_in = 72
battery_capacity = 85
battery_efficiency = 0.95
battery_cost = 0.263

def calculate_energy_demand(T_out):
    return k * abs(T_in - T_out)

def update_battery_state(battery_state, recharge, discharge):
    new_state = battery_state * battery_efficiency + recharge - discharge
    return max(0, min(battery_capacity, new_state))

def calculate_operational_cost(E_grid, E_recharge, E_discharge, P):
    return E_grid * P + E_recharge * P + battery_cost * E_discharge

def calculate_grid_only_cost(E_demand, P):
    return E_demand * P

def mpc_decision_making(current_battery_state, current_time_index, prediction_horizon, max_charge_discharge_rate):
    def cost_function(decision_vars):
        battery_state = current_battery_state
        total_cost = 0
        for i in range(prediction_horizon):
            time_index = current_time_index + i
            if time_index >= len(grid_prices):
                break

            P = get_column_data(grid_prices, 'VALUE').iloc[time_index]
            T_out = get_column_data(temps, 'VALUE').iloc[time_index]
            E_demand = calculate_energy_demand(T_out)

            E_discharge = min(decision_vars[2*i], battery_state, E_demand)
            E_recharge = min(decision_vars[2*i + 1], max_charge_discharge_rate - E_discharge, battery_capacity - battery_state)

            battery_state = update_battery_state(battery_state, E_recharge, E_discharge)
            E_grid = E_demand - E_discharge
            total_cost += calculate_operational_cost(E_grid, E_recharge, E_discharge, P)

        return total_cost

    initial_guess = [max_charge_discharge_rate/2] * prediction_horizon * 2
    bounds = [(0, max_charge_discharge_rate)] * prediction_horizon * 2
    result = minimize(cost_function, initial_guess, bounds=bounds, method='SLSQP')
    return result.x[:2]

os.makedirs('results_new', exist_ok=True)

def run_simulation(prediction_horizon, max_charge_discharge_rate, file_suffix):
    results = []
    battery_state = battery_capacity / 2
    cumulative_cost = 0
    cumulative_grid_only_cost = 0
    for t in tqdm(range(len(grid_prices)), desc=f"Simulating (Horizon: {prediction_horizon}, Discharge Rate: {max_charge_discharge_rate})"):
        E_discharge_decision, E_recharge_decision = mpc_decision_making(battery_state, t, prediction_horizon, max_charge_discharge_rate)
        
        P = get_column_data(grid_prices, 'VALUE').iloc[t]
        T_out = get_column_data(temps, 'VALUE').iloc[t]
        E_demand = calculate_energy_demand(T_out)

        E_discharge = min(E_discharge_decision, battery_state, E_demand)
        E_recharge = min(E_recharge_decision, max_charge_discharge_rate - E_discharge, battery_capacity - battery_state)
        E_grid = E_demand - E_discharge
        battery_state = update_battery_state(battery_state, E_recharge, E_discharge)

        cost_with_battery = calculate_operational_cost(E_grid, E_recharge, E_discharge, P)
        cumulative_cost += cost_with_battery
        grid_only_cost = calculate_grid_only_cost(E_demand, P)
        cumulative_grid_only_cost += grid_only_cost
        savings_with_battery = cumulative_grid_only_cost - cumulative_cost

        results.append({
            'Time': grid_prices.iloc[t]['DATE'],
            'Grid Price': P,
            'Temp': T_out,
            'Energy Demand (kW)': E_demand,
            'Amount of Battery Recharge (kWh)': E_recharge,
            'Amount of Battery Discharge (kWh)': E_discharge,
            'Grid Power (kWh)': E_grid,
            'Total Power': E_discharge + E_grid,
            'Battery Power Stored (kWh)': battery_state,
            'Net Power Change (kWh)': E_recharge - E_discharge,
            'Cost of Time Step with Battery': cost_with_battery,
            'Cumulative Cost with Battery': cumulative_cost,
            'Cost of Time Step without Battery': grid_only_cost,
            'Cumulative Cost without Battery': cumulative_grid_only_cost,
            'Savings with Battery': savings_with_battery
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results_new/with_battery_{file_suffix}.csv', index=False)

prediction_horizons = [2, 3, 4, 5, 6]
discharge_rates = [1, 2, 3, 6, 12]

for horizon in prediction_horizons:
    run_simulation(horizon, 3, f"horizon_{horizon}")

for discharge_rate in discharge_rates:
    run_simulation(3, discharge_rate, f"discharge_{int(discharge_rate)}")

