import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

def read_and_interpolate_csv(input_file, output_file):
    """
    Read CSV with monthly measurements and interpolate to daily data with random noise
    """
    
    print(f"📁 Reading CSV file: {input_file}")
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"✅ Successfully loaded {len(df)} monthly records")
    
    print("📅 Creating date column from year and month...")
    # First, let's check what columns we actually have
    print(f"🔍 Available columns: {list(df.columns)}")
    print(f"📋 First few rows of data:")
    print(df.head())
    
    # Create a date column from year and month (add day=1 for first day of month)
    try:
        df['date'] = pd.to_datetime(df[['y', 'm']].assign(day=1))
    except:
        # Alternative method - create date string first, then convert
        df['date_str'] = df['y'].astype(str) + '-' + df['m'].astype(str).str.zfill(2) + '-01'
        df['date'] = pd.to_datetime(df['date_str'])
        df = df.drop('date_str', axis=1)
        print("✅ Used alternative date creation method")
    
    print("🔄 Sorting data by series, zone, and date...")
    # Sort by date, kvfb, and zone
    df = df.sort_values(['kvfb', 'zone', 'date'])
    
    # Print data overview
    print(f"📊 Data overview:")
    print(f"   • Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    print(f"   • Series (kvfb): {', '.join(df['kvfb'].unique())}")
    print(f"   • Zones: {', '.join(df['zone'].unique())}")
    print(f"   • Unique combinations: {len(df.groupby(['kvfb', 'zone']))}")
    
    print("\n🔄 Starting interpolation process...")
    
    # Create an empty list to store daily data
    daily_data = []
    
    # Process each combination of kvfb and zone
    processed_combinations = 0
    total_combinations = len(df.groupby(['kvfb', 'zone']))
    
    for kvfb in df['kvfb'].unique():
        for zone in df['zone'].unique():
            # Filter data for this combination
            subset = df[(df['kvfb'] == kvfb) & (df['zone'] == zone)].copy()
            
            if len(subset) == 0:
                continue
                
            processed_combinations += 1
            print(f"⚙️  Processing combination {processed_combinations}/{total_combinations}: {kvfb} - {zone} ({len(subset)} months)")
                
            # Sort by date
            subset = subset.sort_values('date')
            
            # Create daily interpolation for each consecutive pair of months
            for i in range(len(subset)):
                current_row = subset.iloc[i]
                current_date = current_row['date']
                current_value = current_row['meas']
                
                # Get number of days in current month
                days_in_month = calendar.monthrange(current_row['y'], current_row['m'])[1]
                
                # If this is not the last row, get next month's data for interpolation
                if i < len(subset) - 1:
                    next_row = subset.iloc[i + 1]
                    next_date = next_row['date']
                    next_value = next_row['meas']
                    
                    # Check if next month is consecutive
                    expected_next = current_date + pd.DateOffset(months=1)
                    if next_date == expected_next:
                        # Interpolate between current and next month
                        # Create daily values using linear interpolation
                        daily_values = np.linspace(current_value, next_value, days_in_month + 1)[:-1]
                    else:
                        # Gap in data, use constant value for current month
                        daily_values = [current_value] * days_in_month
                else:
                    # Last data point, use constant value
                    daily_values = [current_value] * days_in_month
                
                # Generate daily data for current month
                for day in range(1, days_in_month + 1):
                    daily_date = datetime(current_row['y'], current_row['m'], day)
                    base_value = daily_values[day - 1]
                    
                    # Add random error between -2% and +2% (reduce from ±10% to ±2%)
                    noise_factor = np.random.uniform(0.98, 1.02)
                    noisy_value = base_value * noise_factor
                    
                    daily_data.append({
                        'date': daily_date,
                        'y': current_row['y'],
                        'm': current_row['m'],
                        'd': day,
                        'meas': round(noisy_value, 2),
                        'kvfb': kvfb,
                        'zone': zone
                    })
    
    print(f"\n📈 Generated {len(daily_data)} daily records with interpolation and noise")
    
    # Create DataFrame from daily data
    daily_df = pd.DataFrame(daily_data)
    
    print("🔄 Sorting final dataset...")
    # Sort by date, kvfb, and zone
    daily_df = daily_df.sort_values(['date', 'kvfb', 'zone'])
    
    print(f"💾 Saving daily data to: {output_file}")
    # Save to CSV
    daily_df.to_csv(output_file, index=False)
    print("✅ File saved successfully!")
    
    return daily_df

def alternative_interpolation(input_file, output_file):
    """
    Alternative approach using pandas interpolation methods
    """
    
    print(f"📁 [Alternative Method] Reading CSV file: {input_file}")
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"✅ Successfully loaded {len(df)} monthly records")
    
    print("📅 Creating date column...")
    # First check the data structure
    print(f"🔍 Available columns: {list(df.columns)}")
    
    # Create a date column
    try:
        df['date'] = pd.to_datetime(df[['y', 'm']].assign(day=1))
    except:
        # Alternative method
        df['date_str'] = df['y'].astype(str) + '-' + df['m'].astype(str).str.zfill(2) + '-01'
        df['date'] = pd.to_datetime(df['date_str'])
        df = df.drop('date_str', axis=1)
        print("✅ Used alternative date creation method")
    
    # Create an empty list for results
    all_daily_data = []
    
    print("🔄 Processing series-zone combinations...")
    processed = 0
    total_combinations = len(df.groupby(['kvfb', 'zone']))
    
    # Process each combination of kvfb and zone
    for kvfb in df['kvfb'].unique():
        for zone in df['zone'].unique():
            subset = df[(df['kvfb'] == kvfb) & (df['zone'] == zone)].copy()
            
            if len(subset) == 0:
                continue
            
            processed += 1
            print(f"⚙️  [Alt] Processing {processed}/{total_combinations}: {kvfb} - {zone}")
            
            # Sort by date
            subset = subset.sort_values('date')
            
            # Create a complete date range from first to last date
            start_date = subset['date'].min()
            end_date = subset['date'].max()
            
            # Create monthly date range
            monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            
            # Create a complete series with all months
            complete_series = pd.DataFrame({'date': monthly_dates})
            complete_series = complete_series.merge(subset[['date', 'meas']], on='date', how='left')
            
            # Interpolate missing values
            complete_series['meas'] = complete_series['meas'].interpolate(method='linear')
            
            # Now create daily data
            for _, row in complete_series.iterrows():
                current_date = row['date']
                current_value = row['meas']
                
                # Get days in this month
                year = current_date.year
                month = current_date.month
                days_in_month = calendar.monthrange(year, month)[1]
                
                # Generate daily values with slight variation
                for day in range(1, days_in_month + 1):
                    daily_date = datetime(year, month, day)
                    
                    # Add small daily variation (sine wave) plus reduced random noise (±2%)
                    day_factor = 1 + 0.02 * np.sin(2 * np.pi * day / days_in_month)  # Reduced from 0.05 to 0.02
                    noise_factor = np.random.uniform(0.98, 1.02)  # Reduced from ±10% to ±2%
                    daily_value = current_value * day_factor * noise_factor
                    
                    all_daily_data.append({
                        'date': daily_date,
                        'y': year,
                        'm': month,
                        'd': day,
                        'meas': round(daily_value, 2),
                        'kvfb': kvfb,
                        'zone': zone
                    })
    
    print(f"\n📈 [Alt] Generated {len(all_daily_data)} daily records")
    
    # Create DataFrame
    daily_df = pd.DataFrame(all_daily_data)
    daily_df = daily_df.sort_values(['date', 'kvfb', 'zone'])
    
    print(f"💾 [Alt] Saving to: {output_file}")
    # Save to CSV
    daily_df.to_csv(output_file, index=False)
    print("✅ [Alt] File saved successfully!")
    
    return daily_df

# Example usage
if __name__ == "__main__":
    print("🚀 Starting CSV interpolation process...")
    print("=" * 50)
    
    # Set random seed for reproducible results (optional)
    np.random.seed(42)
    print("🎲 Random seed set to 42 for reproducible results")
    
    input_filename = "TimeSeriesMonthlyData.csv"  # Replace with your input file name
    output_filename = "daily_data.csv"   # Output file name
    
    try:
        # Use the main interpolation function
        daily_data = read_and_interpolate_csv(input_filename, output_filename)
        
        print("\n" + "=" * 50)
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"✅ Successfully processed {len(daily_data)} daily records")
        print(f"📅 Date range: {daily_data['date'].min().strftime('%Y-%m-%d')} to {daily_data['date'].max().strftime('%Y-%m-%d')}")
        print(f"📈 Series (kvfb): {', '.join(daily_data['kvfb'].unique())}")
        print(f"🌍 Zones: {', '.join(daily_data['zone'].unique())}")
        
        # Display sample of the data
        print(f"\n📋 Sample of daily data (first 10 rows):")
        print("-" * 40)
        sample_data = daily_data.head(10)[['date', 'meas', 'kvfb', 'zone']]
        for _, row in sample_data.iterrows():
            print(f"📅 {row['date'].strftime('%Y-%m-%d')} | 📊 {row['meas']:8.2f} | 📈 {row['kvfb']} | 🌍 {row['zone']}")
        
        # Show statistics
        print(f"\n📈 Data statistics by series and zone:")
        print("-" * 60)
        summary = daily_data.groupby(['kvfb', 'zone'])['meas'].agg(['count', 'mean', 'min', 'max']).round(2)
        for (kvfb, zone), stats in summary.iterrows():
            print(f"📊 {kvfb}-{zone}: {stats['count']:,} days | Avg: {stats['mean']:8.2f} | Min: {stats['min']:8.2f} | Max: {stats['max']:8.2f}")
        
        print(f"\n🎉 Process completed successfully!")
        print(f"💾 Output saved as: {output_filename}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find input file '{input_filename}'")
        print("📁 Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        import traceback
        print("🔍 Full error details:")
        traceback.print_exc()