import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import optuna

import os
# Ensure offline mode for Hugging Face to avoid internet access issues
# os.environ["HF_HUB_OFFLINE"] = "1"

horizon  = None
val_df   = None
train_df = None
versions = None

# --------------------------------------------------------------------------------
# 2. DEFINE EVAL FUNCTIONS
# --------------------------------------------------------------------------------

# 2A. Custom PyTorch TCN Components
class EstateDataset(Dataset):
    def __init__(self, df, seq_len, horizon):
        self.seq_len = seq_len
        self.horizon = horizon
        self.series = []
        
        for uid, sdf in df.groupby("unique_id"):
            sdf = sdf.sort_values("ds").reset_index(drop=True)
            y_vals = sdf["y"].values
            mask_vals = (~sdf["is_pad"]).astype(np.float32).values  # 1.0 = real, 0.0 = pad
            
            for i in range(len(sdf) - seq_len - horizon + 1):
                x = y_vals[i:i+seq_len]
                y = y_vals[i+seq_len:i+seq_len+horizon]
                m = mask_vals[i+seq_len:i+seq_len+horizon]
                self.series.append((x, y, m))
    
    def __len__(self):
        return len(self.series)
    
    def __getitem__(self, idx):
        x, y, m = self.series[idx]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32), \
               torch.tensor(m, dtype=torch.float32)    
    '''
    def __init__(self, series_list, seq_len, horizon):
        self.samples = []
        for y in series_list:
            for i in range(len(y) - seq_len - horizon + 1):
                self.samples.append((y[i:i+seq_len], y[i+seq_len:i+seq_len+horizon]))
    def __len__(self): return len(self.samples)
    # ensure bias and weights are float32: Torch expects them to match
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32)
    '''
class Chomp1d(nn.Module):
    def __init__(self, chomp): super().__init__(); self.chomp=chomp
    def forward(self, x): return x[:, :, :-self.chomp]

class TemporalBlock(nn.Module):
    def __init__(self, ni, no, kernel, dilation, padding, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ni, no, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(no, no, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(drop),
        )
        self.down = nn.Conv1d(ni, no, 1) if ni!=no else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.net(x) + self.down(x))

# This class defines a Temporal Convolutional Network (TCN) model tailored for sequence-to-multioutput forecasting
# This is a PyTorch implementation of a Temporal Convolutional Network, a deep learning architecture 
# designed for modeling sequential data — especially time series. 
# Unlike RNNs or LSTMs, TCNs use dilated causal convolutions to capture long-range dependencies efficiently.
class TCNModel(nn.Module):
    def __init__(self, seq_len, channels, dropout, horizon):
        super().__init__()
        layers = []
        for i, c in enumerate(channels):
            dilation = 2**i
            pad      = (3-1)*dilation
            in_ch    = 1 if i==0 else channels[i-1]
            layers.append(TemporalBlock(in_ch, c, kernel=3,
                                        dilation=dilation, padding=pad,
                                        drop=dropout))
        self.tcn    = nn.Sequential(*layers)
        self.linear = nn.Linear(channels[-1], horizon)
    def forward(self, x):
        out = self.tcn(x)[:, :, -1]
        return self.linear(out)

def get_series(df, uid):
    return df[df["unique_id"]==uid].sort_values("ds")["y"].values

def eval_pytorch_tcn(trial):
    print("🧪 Evaluating PyTorch TCN model...")

    # hyperparameters to tune
    seq_len   = trial.suggest_int("pt_seq_len", 30, 180)
    lr        = trial.suggest_float("pt_lr", 1e-4, 1e-2, log=True)
    dropout   = trial.suggest_float("pt_dropout", 0.0, 0.3)
    n_layers  = trial.suggest_int("pt_n_layers", 1, 3)
    channels  = [ trial.suggest_int(f"pt_ch{i}", 16, 128, log=True)
                  for i in range(n_layers) ]

    print("📐 Building dataset and dataloader...")
    train_series = [ get_series(train_df, v) for v in versions ]
    ds            = EstateDataset(train_series, seq_len, horizon)
    dl            = DataLoader(ds, batch_size=64, shuffle=True)

    print("🧠 Initializing model and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TCNModel(seq_len, channels, dropout, horizon).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn= nn.L1Loss()

    print("🏋️ Training model...")
    model.train()

    for _ in range(10):   # epochs
        for x_b, y_b, m_b in dl:       # <-- now 3 items: inputs, targets, mask
            x_b = x_b.to(device).unsqueeze(1)    # add channel dim
            y_b = y_b.to(device)
            m_b = m_b.to(device)

            opt.zero_grad()
            out = model(x_b)

            # 🔹 Masked MAE loss: ignore padded targets
            loss = (torch.abs(out - y_b) * m_b).sum() / m_b.sum()

            loss.backward()
            opt.step()

    print("🔍 Forecasting and evaluating MAE...")
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for v in versions:
            series = get_series(train_df, v)   # full real training history
            inp = torch.tensor(series[-seq_len:], dtype=torch.float32)\
                    .unsqueeze(0).unsqueeze(0).to(device)
            p = model(inp).cpu().numpy().flatten()

            preds.extend(p.tolist())
            actuals.extend(get_series(val_df, v).tolist())

    mae = mean_absolute_error(actuals, preds)
    return mae

# 2B. NeuralForecast TCN
from neuralforecast import NeuralForecast
from neuralforecast.models import TCN as NFTCN

def eval_nf_tcn(trial):
    print("🧪 Evaluating NeuralForecast TCN model...")
    input_size = trial.suggest_int("nf_input_size", 30, 180)
    nf = NeuralForecast(
        models=[ NFTCN(input_size=input_size, h=horizon) ],
        freq="D"
    )
    print("📈 Fitting NeuralForecast model...")
    print("🏋️ Training model...")
    nf.fit(train_df)
    print("🔍 Forecasting...")
    pred = nf.predict()
    merged = val_df.merge(pred, on=["ds","unique_id"])
    print("📊 Calculating MAE...")
    return mean_absolute_error(merged["y"], merged["TCN"])

# 2C. CHRONOS Fine-tuning
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import pandas as pd
from sklearn.metrics import mean_absolute_error

def eval_chronos(trial):
    print("🧪 Evaluating Chronos model...")
    # 1️⃣ Pick a CPU‐compatible Chronos-Bolt preset
    cpu_presets = ["bolt_tiny", "bolt_mini", "bolt_small", "bolt_base"]
    preset     = trial.suggest_categorical("chronos_preset", cpu_presets)
    print(f"⚙️ Using Chronos preset: {preset}")

    # 2️⃣ Rename your DataFrames to Chronos’s expected schema
    train_ch = train_df.rename(
        columns={"ds": "timestamp", "y": "target", "unique_id": "item_id"}
    )
    val_ch   = val_df.rename(
        columns={"ds": "timestamp", "y": "target", "unique_id": "item_id"}
    )
    print("📐 Preparing training & validation data...")

    # 3️⃣ Build TimeSeriesDataFrame for training only
    train_ts = TimeSeriesDataFrame.from_data_frame(train_ch)

    # 4️⃣ Fit the Chronos model
    print("📈 Fitting Chronos model...")
    predictor = TimeSeriesPredictor(
        path=f"chronos_model_{trial.number}",
        prediction_length=horizon,
    )
    predictor.fit(train_ts, presets=preset)

    # 5️⃣ Generate horizon‐ahead forecasts
    print("🔮 Generating forecasts...")
    fcst = predictor.predict(train_ts).reset_index()

    # 6️⃣ Show quick samples for debugging
    print("🔎 Forecast sample:")
    print(fcst.head())

    print("🧪 Validation sample:")
    print(val_ch.head())

    # 7️⃣ Filter forecasts to your validation window
    fcst["timestamp"] = pd.to_datetime(fcst["timestamp"])
    val_ch["timestamp"] = pd.to_datetime(val_ch["timestamp"])
    mask = (
        (fcst["timestamp"] >= val_ch["timestamp"].min()) &
        (fcst["timestamp"] <= val_ch["timestamp"].max())
    )
    val_preds   = fcst[mask]
    missing_ids = set(val_ch["item_id"]) - set(val_preds["item_id"])
    if missing_ids:
        print(f"🚫 Missing forecasts for: {missing_ids}")

    # 8️⃣ Merge & score
    merged = val_ch.merge(val_preds, on=["item_id", "timestamp"], how="inner")
    if merged.empty:
        print("⚠️ Merged DataFrame is empty — no overlap between forecast and validation.")
        return float("inf")

    print("📊 Calculating MAE...")
    return mean_absolute_error(merged["target"], merged["mean"])

# --------------------------------------------------------------------------------
# 3. OPTUNA STUDY
# --------------------------------------------------------------------------------
# model selection and hyperparameter optimization loop, powered by Optuna: see also https://optuna.readthedocs.io/en/stable/
# This is the objective function that Optuna calls repeatedly to evaluate different combinations of parameters. 
def objective(trial):
    # Optuna randomly selects one of the three model types:
    # •	 → my custom PyTorch Temporal Convolutional Network
    # •	 → NeuralForecast’s TCN implementation
    # •	 → AutoGluon’s Chronos mode
    model_choice = trial.suggest_categorical("model", ["pt_tcn", "nf_tcn", "chronos"])
    print(f"🧠 Trial started with model: {model_choice}")
    if model_choice == "pt_tcn":
        return eval_pytorch_tcn(trial)
    if model_choice == "nf_tcn":
        return eval_nf_tcn(trial)
    return eval_chronos(trial)

def main():
    # --------------------------------------------------------------------------------
    # 1. LOAD & PREPROCESS
    # --------------------------------------------------------------------------------

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

    print("✂️ Splitting into training and validation sets...")
    horizon  = 30
    max_date = df["ds"].max()
    train_df = df[df["ds"] <= (max_date - pd.Timedelta(days=horizon))]
    val_df   = df[(df["ds"] >  (max_date - pd.Timedelta(days=horizon))) &
                (df["ds"] <= max_date)]

    versions = df["unique_id"].unique().tolist()
    print(f"🧬 Found {len(versions)} unique versions.")

    print("🚀 Starting Optuna study...")
    # Creates a new optimization study. See also https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
    # Sets the optimization direction to "minimize", which tells Optuna to search for the lowest possible MAE.
    ## study = optuna.create_study(direction="minimize")
    # Create a study backed by SQLite
    storage_url = "sqlite:///my_study.db"
    study = optuna.create_study(
        direction="minimize",
        study_name="forecast_study",
        storage=storage_url,
        load_if_exists=True   # append if rerun
    )

    # •	Runs up to XX trials or until HH hours has passed.
    # •	Each trial randomly selects a model and hyperparameters, then evaluates performance.
    study.optimize(objective, n_trials=50, timeout=36000) ## XX trials, HH hours

    print("🏆 Best trial parameters:", study.best_trial.params)
    print("📉 Best MAE:", study.best_value)

if __name__ == "__main__":
    main()