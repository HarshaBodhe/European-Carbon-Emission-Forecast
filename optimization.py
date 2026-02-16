import pulp

# 🎯 Define the optimization problem
model = pulp.LpProblem("Urban_Carbon_Optimization", pulp.LpMaximize)

# Decision variables: Investment (in million €) in each sector
renewable_energy = pulp.LpVariable('Renewable_Energy', lowBound=0)
public_transport = pulp.LpVariable('Public_Transport', lowBound=0)
green_buildings = pulp.LpVariable('Green_Buildings', lowBound=0)
waste_management = pulp.LpVariable('Waste_Management', lowBound=0)

# 📈 Objective: Maximize CO₂ reduction potential (in tons)
# Assume per € million invested, reduction potential:
model += (
    0.9 * renewable_energy +
    0.7 * public_transport +
    0.5 * green_buildings +
    0.4 * waste_management
), "Total_CO2_Reduction"

# 💰 Budget constraint (in € million)
model += renewable_energy + public_transport + green_buildings + waste_management <= 100, "BudgetConstraint"

# 🌍 Policy constraints
model += renewable_energy >= 10, "MinimumRenewableInvestment"
model += public_transport <= 40, "TransportCap"
model += green_buildings >= 5, "BuildingMinimum"

# 🔧 Solve the model
model.solve()

# 📊 Print the results
print("=== Optimal Carbon Reduction Strategy ===")
for variable in model.variables():
    print(f"{variable.name}: €{variable.varValue:.2f} million")

print(f"\n✅ Total CO₂ Reduction Potential: {pulp.value(model.objective):.2f} tons")
print(f"✅ Total Budget Used: {sum(v.varValue for v in model.variables()):.2f} million €")
