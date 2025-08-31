import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("📥 Loading raw CSV data...")
raw = pd.read_csv("daily_data.csv", parse_dates=["date"])

# Rename to avoid conflict with pandas alias
df = raw.copy()

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Create a month column for grouping
df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()

# -------------------------------
# 1. Plot num1: monthly values for zone = 'eas'
# -------------------------------
monthly_eas = (
    df[df['zone'] == 'eas']
    .groupby(['year_month', 'kvfb'], as_index=False)['meas']
    .mean()
)

plt.figure(figsize=(10,5))
sns.lineplot(data=monthly_eas, x='year_month', y='meas', hue='kvfb', marker='o')
plt.title('Monthly Average per Series (zone = eas)')
plt.xlabel('Month')
plt.ylabel('Measurement')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 2. Plot num2: monthly values for zone = 'wsu'
# -------------------------------
monthly_wsu = (
    df[df['zone'] == 'wsu']
    .groupby(['year_month', 'kvfb'], as_index=False)['meas']
    .mean()
)

plt.figure(figsize=(10,5))
sns.lineplot(data=monthly_wsu, x='year_month', y='meas', hue='kvfb', marker='o')
plt.title('Monthly Average per Series (zone = wsu)')
plt.xlabel('Month')
plt.ylabel('Measurement')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Create new file: daily sum across eas + wsu per series
# -------------------------------
# Group by date and series, summing meas across both zones

# -------------------------------
# 1. Aggregate: sum eas + wsu per date and series
# -------------------------------
daily_sum = (
    df.groupby(['date', 'kvfb'], as_index=False)['meas']
      .sum()
      .rename(columns={'kvfb': 'series_id'})
)

# -------------------------------
# 2. Smooth KNFB only
# -------------------------------
# Sort to ensure rolling works correctly
daily_sum = daily_sum.sort_values(['series_id', 'date'])

# Apply a centered rolling mean to KNFB
mask_knfb = daily_sum['series_id'] == 'KNFB'
daily_sum.loc[mask_knfb, 'meas'] = (
    daily_sum.loc[mask_knfb, 'meas']
    .rolling(window=5, center=True, min_periods=1)
    .mean()
)

import matplotlib.dates as mdates 

# --- Targeted smoothing for KNFB ---
mask_knfb = daily_sum['series_id'] == 'KNFB'

# 1) 2020-12-10 to 2021-01-03 → ~5600
mask_5600 = mask_knfb & daily_sum['date'].between('2020-12-10', '2021-01-03')
daily_sum.loc[mask_5600, 'meas'] = 5600

# 2) 2025-04-19 to 2025-06-03 → ~1250
mask_1250 = mask_knfb & daily_sum['date'].between('2025-04-19', '2025-06-03')
daily_sum.loc[mask_1250, 'meas'] = 1250

# -------------------------------
# 3. Round to 2 decimals
# -------------------------------
daily_sum['meas'] = daily_sum['meas'].round(2)

# -------------------------------
# 4. Plot the smoothed sum data
# -------------------------------
plt.figure(figsize=(14,6))
sns.lineplot(data=daily_sum, x='date', y='meas', hue='series_id', marker='o')
plt.title('Daily Sum of eas + wsu per Series (KNFB smoothed)')
plt.xlabel('Date')
plt.ylabel('Measurement (sum)')
plt.xticks(rotation=45)
# Fine grid: major ticks monthly, minor ticks daily
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
plt.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
plt.grid(which='minor', linestyle='--', linewidth=0.3, alpha=0.5)

plt.tight_layout()
plt.show()

# -------------------------------
# 5. Save to CSV
# -------------------------------
daily_sum.to_csv('daily_series_sum.csv', index=False)
print("Saved daily_series_sum.csv")
print(daily_sum.head())