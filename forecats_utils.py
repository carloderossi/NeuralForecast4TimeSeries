import pandas as pd
import optuna

def load_data():
    print("📥 Loading raw CSV data...")
    raw = pd.read_csv("daily_data.csv", parse_dates=["date"])

    print("🧮 Aggregating measurements by date and version...")
    df = (
        raw
        .groupby(["date", "kvfb"], as_index=False)["meas"]
        .sum()
        .rename(columns={
            "date":     "ds",
            "kvfb":     "unique_id",
            "meas":     "y",
        })
    )

    print("📊 Sorting data chronologically...")
    df = df.sort_values(["unique_id", "ds"])

    # 🩹 Apply masked padding for late-start series
    print("🩹 Applying masked padding for late-start series...")
    earliest_start = df["ds"].min()
    df = pad_series_with_mask(df, start_date=earliest_start)

    print("✂️ Splitting into training and validation sets...")
    horizon  = 30
    max_date = df["ds"].max()
    train_df = df[df["ds"] <= (max_date - pd.Timedelta(days=horizon))]
    val_df   = df[(df["ds"] >  (max_date - pd.Timedelta(days=horizon))) &
                  (df["ds"] <= max_date)]

    versions = df["unique_id"].unique().tolist()
    print(f"🧬 Found {len(versions)} unique versions.")

    return df, train_df, val_df, versions

def load_study_results():
    """Load and display Optuna study results."""
    storage_url = "sqlite:///my_study.db"
    
    study = optuna.load_study(
        study_name="forecast_study",
        storage=storage_url
    )
    
    print("🏆 Best trial parameters:", study.best_trial.params)
    print("📉 Best MAE:", study.best_value)
    
    # Access full history if you want
    for t in study.trials:
        print(f"Trial {t.number}: value={t.value}, params={t.params}")
    
    return study.best_trial.params

def prepare_data():
    """Load and prepare data for forecasting."""
    df, train_df, val_df, versions = load_data()
    
    horizon = val_df["ds"].nunique()
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    versions = full_df["unique_id"].unique()
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'full_df': full_df,
        'versions': versions,
        'horizon': horizon
    }

def pad_series_with_mask(df, start_date):
    """Pad each unique_id back to start_date with zero values and mask flag."""
    start_date = pd.to_datetime(start_date)
    padded_parts = []
    
    for uid, sdf in df.groupby("unique_id"):
        sdf = sdf.sort_values("ds").copy()
        first_date = sdf["ds"].min()
        
        if first_date > start_date:
            filler_dates = pd.date_range(start_date, first_date - pd.Timedelta(days=1), freq="D")
            filler = pd.DataFrame({
                "unique_id": uid,
                "ds": filler_dates,
                "y": 0.0,
                "is_pad": True
            })
            sdf["is_pad"] = False
            sdf = pd.concat([filler, sdf], ignore_index=True)
        else:
            sdf["is_pad"] = False
        
        padded_parts.append(sdf)
    
    return pd.concat(padded_parts).reset_index(drop=True)