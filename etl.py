import os
import time
import requests
import pandas as pd
import numpy as np

OUT_DIR = "."
START = "2000-01-01"
END = "2023-12-01"

COUNTRIES = [
    "Germany", "France", "United Kingdom", "Italy", "Spain",
    "Poland", "Netherlands", "Belgium", "Sweden", "Austria",
    "Denmark", "Portugal", "Greece", "Czechia", "Hungary",
    "Ireland", "Finland", "Romania", "Bulgaria", "Slovakia"
]

ISO3_MAP = {
    "Germany": "DEU", "France":"FRA", "United Kingdom":"GBR", "Italy":"ITA", "Spain":"ESP",
    "Poland":"POL", "Netherlands":"NLD", "Belgium":"BEL", "Sweden":"SWE", "Austria":"AUT",
    "Denmark":"DNK", "Portugal":"PRT", "Greece":"GRC", "Czechia":"CZE", "Hungary":"HUN",
    "Ireland":"IRL", "Finland":"FIN", "Romania":"ROU", "Bulgaria":"BGR", "Slovakia":"SVK"
}

def fetch_worldbank_gdp_per_capita(iso3, start=2000, end=2023):
    url = f"http://api.worldbank.org/v2/country/{iso3}/indicator/NY.GDP.PCAP.CD"
    params = {"date": f"{start}:{end}", "format": "json", "per_page": 500}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"WorldBank: non-200 for {iso3}: {r.status_code}")
            return pd.DataFrame()
        js = r.json()
        if not isinstance(js, list) or len(js) < 2:
            return pd.DataFrame()
        records = []
        for rec in js[1]:
            year = int(rec["date"])
            val = rec["value"]
            records.append({"year": year, "gdp_per_capita": val})
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
        return df[["date", "gdp_per_capita"]]
    except Exception as e:
        print(f"WorldBank API error for {iso3}: {e}")
        return pd.DataFrame()

def generate_country_timeseries(country_name, start=START, end=END, use_worldbank=True):
    dates = pd.date_range(start, end, freq="ME")
    n = len(dates)

    base_co2_map = {"Germany": 9, "France": 6, "United Kingdom": 7.5, "Italy": 6.8, "Spain": 5.5}
    base = base_co2_map.get(country_name, 5.0)

    years = dates.year - dates.year.min()
    co2 = np.clip(base + 0.01 * years + 0.2 * np.sin(2*np.pi*dates.month/12) +
                  np.random.normal(0, 0.25, n), 0, None)
    gdp = 20000 + 400 * years + np.random.normal(0, 1500, n)
    temp = 8 + 0.03 * years + np.random.normal(0, 1.2, n)
    renew = np.clip(0.10 + 0.01 * years + np.random.normal(0, 0.01, n), 0, 1)

    df = pd.DataFrame({
        "date": dates,
        "country": country_name,
        "co2_per_capita": co2,
        "gdp_per_capita": gdp,
        "temp_avg": temp,
        "renewable_share": renew
    })

    # ✅ FIXED — safe World Bank integration
    if use_worldbank and country_name in ISO3_MAP:
        iso3 = ISO3_MAP[country_name]
        wb = fetch_worldbank_gdp_per_capita(iso3, start=dates.year.min(), end=dates.year.max())

        if not wb.empty and "gdp_per_capita" in wb.columns:
            df = pd.merge_asof(
                df.sort_values("date"),
                wb.sort_values("date"),
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta('400d')
            )

            if "gdp_per_capita_y" in df.columns:
                df["gdp_per_capita"] = df["gdp_per_capita_y"].fillna(df["gdp_per_capita_x"])

            df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")
        else:
            print(f"World Bank data missing for {country_name}. Using synthetic GDP.")

    return df

def generate_berlin_timeseries(start=START, end=END):
    """Generate Berlin-specific timeseries data"""
    dates = pd.date_range(start, end, freq="ME")
    n = len(dates)
    
    # Berlin-specific parameters (urban area, lower CO2 per capita than Germany average)
    base_co2 = 5.0  # Lower than Germany average (9) since it's a city
    years = dates.year - dates.year.min()
    
    co2 = np.clip(base_co2 + 0.01 * years + 0.2 * np.sin(2*np.pi*dates.month/12) +
                  np.random.normal(0, 0.25, n), 0, None)
    # Berlin has higher GDP per capita than Germany average
    gdp = 25000 + 450 * years + np.random.normal(0, 1500, n)
    temp = 9 + 0.03 * years + np.random.normal(0, 1.2, n)  # Slightly warmer
    renew = np.clip(0.12 + 0.01 * years + np.random.normal(0, 0.01, n), 0, 1)  # Higher renewable share
    
    df = pd.DataFrame({
        "date": dates,
        "country": "Berlin",
        "co2_per_capita": co2,
        "gdp_per_capita": gdp,
        "temp_avg": temp,
        "renewable_share": renew
    })
    
    # Use Germany's World Bank data as proxy for Berlin
    wb = fetch_worldbank_gdp_per_capita("DEU", start=dates.year.min(), end=dates.year.max())
    if not wb.empty and "gdp_per_capita" in wb.columns:
        df = pd.merge_asof(
            df.sort_values("date"),
            wb.sort_values("date"),
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta('400d')
        )
        
        if "gdp_per_capita_y" in df.columns:
            # Scale up Germany GDP for Berlin (urban premium)
            df["gdp_per_capita"] = (df["gdp_per_capita_y"].fillna(df["gdp_per_capita_x"]) * 1.2)
        
        df = df.drop(columns=[c for c in df.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")
    
    return df

def main():
    all_dfs = []
    print("Generating datasets for:", ", ".join(COUNTRIES))
    print("Also generating Berlin dataset...")

    # Generate all country datasets
    for c in COUNTRIES:
        print(" -", c)
        df = generate_country_timeseries(c, use_worldbank=True)
        df.to_csv(f"{c.lower().replace(' ','_')}_timeseries.csv", index=False)
        all_dfs.append(df)
        time.sleep(0.2)

    # Generate Berlin dataset
    print(" - Berlin")
    df_berlin = generate_berlin_timeseries()
    df_berlin.to_csv("berlin_timeseries.csv", index=False)
    print("✅ Generated berlin_timeseries.csv")

    # Combine all country data (excluding Berlin from combined file)
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv("europe_timeseries.csv", index=False)
    print("\n✅ Saved combined file: europe_timeseries.csv")
    print("✅ Shape:", df_all.shape)
    print("\n✅ All datasets generated successfully!")

if __name__ == "__main__":
    main()