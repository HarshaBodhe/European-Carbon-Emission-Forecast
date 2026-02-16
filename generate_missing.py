import pandas as pd
import numpy as np

START = "2000-01-01"
END = "2023-12-01"

dates = pd.date_range(START, END, freq="ME")
years = dates.year - dates.year.min()

def generate(country):
    df = pd.DataFrame({
        "date": dates,
        "country": country,
        "co2_per_capita": 5.0 + 0.01*years + np.random.normal(0,0.2,len(dates)),
        "gdp_per_capita": 18000 + 350*years + np.random.normal(0,1400,len(dates)),
        "temp_avg": 8 + 0.04*years + np.random.normal(0,1,len(dates)),
        "renewable_share": np.clip(0.1 + 0.005*years + np.random.normal(0,0.01,len(dates)), 0,1)
    })

    df.to_csv(f"{country.lower()}_timeseries.csv", index=False)
    print("Generated:", country)

generate("bulgaria")
generate("slovakia")
import pandas as pd
import numpy as np

START = "2000-01-01"
END = "2023-12-01"

def generate_slovakia():
    dates = pd.date_range(START, END, freq="ME")
    years = dates.year - dates.year.min()

    df = pd.DataFrame({
        "date": dates,
        "country": "Slovakia",
        "co2_per_capita": np.clip(5 + 0.01*years + np.random.normal(0, 0.3, len(dates)), 0, None),
        "gdp_per_capita": 18000 + 350*years + np.random.normal(0, 1500, len(dates)),
        "temp_avg": 8 + 0.03*years + np.random.normal(0, 1, len(dates)),
        "renewable_share": np.clip(0.08 + 0.01*years + np.random.normal(0, 0.01, len(dates)), 0, 1)
    })

    df.to_csv("slovakia_timeseries.csv", index=False)
    print("✅ Created slovakia_timeseries.csv")

generate_slovakia()

