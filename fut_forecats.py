import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from forecats_utils import load_data, load_study_results, prepare_data

# Import model implementations
from nf_tcn import NeuralForecastTCN
from pt_tcn import PyTorchTCN

def aggregate_to_monthly(df, value_col, timestamp_col):
    """Convert daily data to monthly averages."""
    df = df.copy()
    df["month"] = pd.to_datetime(df[timestamp_col]).dt.to_period("M").dt.to_timestamp()
    return df.groupby(["unique_id", "month"])[value_col].mean().reset_index()

def plot_results(monthly_act, monthly_pred, versions):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    
    # Plot actuals
    for uid in versions:
        df_a = monthly_act[monthly_act.unique_id == uid]
        plt.plot(df_a["month"], df_a["y"], marker="o", label=f"{uid} actual")
    
    # Plot predictions
    for _, r in monthly_pred.iterrows():
        plt.scatter(r["month"], r["predicted"], 
                   s=120, marker="X", color="black", zorder=5)
        plt.text(r["month"], r["predicted"] + 2,
                f"{r['predicted']:.1f}", ha="center")
    
    plt.title("Monthly Actuals + Hold-Out Predictions")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_results_with_future(monthly_act, monthly_pred, monthly_fut, versions):
    """Plot actuals, hold‑out predictions, and 3‑month forecast with fade on forecast."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Plot actual historicals
    for uid in versions:
        df_a = monthly_act[monthly_act.unique_id == uid]
        plt.plot(df_a["month"], df_a["y"], marker="o", label=f"{uid} actual")

    # Plot hold‑out predictions
    for _, r in monthly_pred.iterrows():
        plt.scatter(r["month"], r["predicted"],
                    s=120, marker="X", color="black", zorder=5)
        plt.text(r["month"], r["predicted"] + 2,
                 f"{r['predicted']:.1f}", ha="center")

    # Plot future forecast with faded colour
    for uid in versions:
        df_f = monthly_fut[monthly_fut.unique_id == uid]
        plt.plot(df_f["month"], df_f["future_pred"],
                 marker="s", linestyle="--",
                 # color="tab:orange", alpha=0.5,  # lighter + transparent
                 label=f"{uid} forecast")

    plt.title("Monthly Actuals + Hold-Out Predictions + 3-Month Forecast")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    # Load study results and data
    print("🔍 Loading Optuna study results...")
    best_params = load_study_results()
    print("📊 Preparing data...")
    data = prepare_data()
    
    best_model = best_params["model"]
    print(f"🏆 Best model: {best_model} with params {best_params}")
    
    # Initialize appropriate forecaster
    if best_model == "nf_tcn":
        forecaster = NeuralForecastTCN(best_params)
        print("Using NeuralForecast TCN model.")
    elif best_model == "pt_tcn":
        forecaster = PyTorchTCN(best_params)
        print("Using PyTorch TCN model.")
    else:
        raise ValueError(f"Unknown best model: {best_model}")
    
    # Generate predictions
    pred_val, future_df = forecaster.forecast(data)
    
    # Aggregate to monthly
    monthly_act = aggregate_to_monthly(
        data['full_df'], 'y', 'ds'
    )
    monthly_pred = aggregate_to_monthly(
        pred_val, 'predicted', 'timestamp'
    )
    monthly_fut = aggregate_to_monthly(
        future_df, 'future_pred', 'timestamp'
    )
    
    # Plot results
    # plot_results(monthly_act, monthly_pred, data['versions'])
    
    # Print future forecasts
    print("\n🚀 Next Months Forecast (monthly avg):")
    print(monthly_fut.to_string(index=False))


    plot_results_with_future(monthly_act, monthly_pred, monthly_fut, data['versions'])

if __name__ == "__main__":
    main()