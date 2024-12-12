import pandas as pd
import numpy as np

# Loading data from the CSV file
file_path = 'C:/Users/sacha/Downloads/Optimi4.csv'
data = pd.read_csv(file_path, sep=';')

print("Loaded data:")
print(data)

# Convert data to numpy arrays
supply = data['supply'].astype(int).values
demand = data['demand'].astype(int).values
cost = data[['Cost_1', 'Cost_2', 'Cost_3']].astype(float).values

# Northwest Corner Method
def northwest_corner_method(supply, demand, cost):
    allocation = np.zeros_like(cost)
    i, j = 0, 0
    while i < len(supply) and j < len(demand):
        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]
        if supply[i] == 0:
            i += 1
        if demand[j] == 0:
            j += 1
    return allocation

# Minimum Cost Method
def minimum_cost_method(supply, demand, cost):
    allocation = np.zeros_like(cost)
    temp_cost = cost.copy()
    while np.any(supply > 0) and np.any(demand > 0):
        min_cost_index = np.unravel_index(np.argmin(temp_cost), temp_cost.shape)
        i, j = min_cost_index[0], min_cost_index[1]
        allocation[i, j] = min(supply[i], demand[j])
        supply[i] -= allocation[i, j]
        demand[j] -= allocation[i, j]
        temp_cost[i, j] = np.inf
    return allocation

# Minimum Row Cost Method
def minimum_row_cost_method(supply, demand, cost):
    allocation = np.zeros_like(cost)
    while np.any(supply > 0) and np.any(demand > 0):
        row_costs = np.min(cost, axis=1)
        row_index = np.argmin(row_costs)
        col_index = np.argmin(cost[row_index])
        allocation[row_index, col_index] = min(supply[row_index], demand[col_index])
        supply[row_index] -= allocation[row_index, col_index]
        demand[col_index] -= allocation[row_index, col_index]
        cost[row_index, col_index] = np.inf
    return allocation

# Vogel's Approximation Method
def vogels_method(costs, supply, demand):
    """
    Implements Vogel's Approximation Method (VAM) for transportation problems.
    """
    rows, cols = costs.shape
    allocation = np.zeros((rows, cols))

    supply_left = supply.copy()
    demand_left = demand.copy()

    supply_points = list(range(rows))
    demand_points = list(range(cols))

    while supply_points and demand_points:
        penalties = []

        # Calculate row penalties
        for i in supply_points:
            row_costs = [costs[i][j] for j in demand_points]
            if len(row_costs) >= 2:
                sorted_costs = sorted(row_costs)
                penalties.append((sorted_costs[1] - sorted_costs[0], i, 'row'))
            elif len(row_costs) == 1:
                penalties.append((row_costs[0], i, 'row'))

        # Calculate column penalties
        for j in demand_points:
            col_costs = [costs[i][j] for i in supply_points]
            if len(col_costs) >= 2:
                sorted_costs = sorted(col_costs)
                penalties.append((sorted_costs[1] - sorted_costs[0], j, 'col'))
            elif len(col_costs) == 1:
                penalties.append((col_costs[0], j, 'col'))

        if not penalties:
            break

        penalty, index, kind = max(penalties, key=lambda x: x[0])

        if kind == 'row':
            i = index
            j = demand_points[np.argmin([costs[i][j] for j in demand_points])]
        else:
            j = index
            i = supply_points[np.argmin([costs[i][j] for i in supply_points])]

        alloc = min(supply_left[i], demand_left[j])
        allocation[i, j] = alloc
        supply_left[i] -= alloc
        demand_left[j] -= alloc

        if supply_left[i] < 1e-10:
            supply_points.remove(i)
        if demand_left[j] < 1e-10:
            demand_points.remove(j)

    return allocation

# Applying the different methods
print("\n--- Northwest Corner Method ---")
nw_allocation = northwest_corner_method(supply.copy(), demand.copy(), cost.copy())
print("Allocation Matrix:\n", nw_allocation)

print("\n--- Minimum Cost Method ---")
min_cost_allocation = minimum_cost_method(supply.copy(), demand.copy(), cost.copy())
print("Allocation Matrix:\n", min_cost_allocation)

print("\n--- Minimum Row Cost Method ---")
min_row_cost_allocation = minimum_row_cost_method(supply.copy(), demand.copy(), cost.copy())
print("Allocation Matrix:\n", min_row_cost_allocation)

print("\n--- Vogel's Approximation Method ---")
vogel_allocation = vogels_method(supply.copy(), demand.copy(), cost.copy())
print("Allocation Matrix:\n", vogel_allocation)
