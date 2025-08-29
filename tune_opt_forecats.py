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

import random
# from The Hitchhiker’s Guide to the Galaxy, where 42 is “the answer to life, the universe, and everything.”
SEED = 42

# 1. Python's own RNG
random.seed(SEED)

# 2. NumPy's RNG
np.random.seed(SEED)

# 3. PyTorch's RNG
torch.manual_seed(SEED)

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

            # If 'is_pad' exists, use it; otherwise create a False mask of same length
            if "is_pad" in sdf.columns:
                mask_vals = (~sdf["is_pad"]).astype(np.float32).values
            else:
                mask_vals = np.ones(len(sdf), dtype=np.float32)  # all real, no padding

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

def get_series_df(df, uid):
    """Return the full DataFrame slice for a given uid."""
    return df[df["unique_id"] == uid][["unique_id", "ds", "y"]].copy()

def get_series_y(df, uid):
    """Return only the y‑values as a 1‑D NumPy array."""
    return df[df["unique_id"] == uid]["y"].values  

def eval_pytorch_tcn(trial):
    global train_df, val_df, versions, horizon  # tell Python to use the globals

    print("🧪 Evaluating PyTorch TCN model...")

    # hyperparameters to tune
    seq_len   = trial.suggest_int("pt_seq_len", 30, 180)
    lr        = trial.suggest_float("pt_lr", 1e-4, 1e-2, log=True)
    dropout   = trial.suggest_float("pt_dropout", 0.0, 0.3)
    n_layers  = trial.suggest_int("pt_n_layers", 1, 3)
    channels  = [ trial.suggest_int(f"pt_ch{i}", 16, 128, log=True)
                  for i in range(n_layers) ]

    print("📐 Building dataset and dataloader...")
    #### train_series = [ get_series(train_df, v) for v in versions ]
    ##### ds = EstateDataset(train_series, seq_len, horizon)
    train_series = pd.concat([get_series_df(train_df, v) for v in versions])
    ds = EstateDataset(train_series, seq_len, horizon)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

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
            # ⬇️ Only take the y‑column values for model input
            series_y = get_series_y(train_df, v)
            #inp = torch.tensor(series_y[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            inp = torch.tensor(series_y[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            p = model(inp).cpu().numpy().flatten()
            preds.extend(p.tolist())

            # Same here for actuals — just the target values
            actual_y = get_series_y(val_df, v)
            actuals.extend(actual_y.tolist())

    mae = mean_absolute_error(actuals, preds)
    return mae

# 2B. NeuralForecast TCN
from neuralforecast import NeuralForecast
from neuralforecast.models import TCN as NFTCN

def eval_nf_tcn(trial):
    global train_df, val_df, versions, horizon

    print("🧪 Evaluating NeuralForecast TCN model...")

    # Hyperparameters
    input_size    = trial.suggest_int("nf_input_size", 30, 180)
    num_blocks    = trial.suggest_int("nf_num_blocks", 1, 5)
    num_layers    = trial.suggest_int("nf_num_layers", 1, 4)
    num_filters   = trial.suggest_categorical("nf_num_filters", [16, 32, 64, 128])
    kernel_size   = trial.suggest_int("nf_kernel_size", 2, 5)
    dilation_base = trial.suggest_int("nf_dilation_base", 2, 4)
    dropout       = trial.suggest_float("nf_dropout", 0.0, 0.5)
    lr            = trial.suggest_loguniform("nf_lr", 1e-4, 1e-2)
    batch_size    = trial.suggest_categorical("nf_batch_size", [32, 64, 128])

    # Model
    print("📐 Building NeuralForecast TCN model...")
    tcn_model = NFTCN(
        input_size=input_size,
        h=horizon,
        kernel_size=kernel_size,
        batch_size=batch_size
    )

    nf = NeuralForecast(models=[tcn_model], freq="D")

    # Fit
    print(f"🏋️ Training with lr={lr} bs={batch_size}…")
    nf.fit(df=train_df.copy(), verbose=False)

    # Predict
    print("🔍 Forecasting…")
    preds = nf.predict(df=val_df.copy())

    # Normalise join keys on copies
    val_df_c = val_df.copy()
    preds_c  = preds.copy()

    val_df_c["unique_id"] = val_df_c["unique_id"].astype(str)
    preds_c["unique_id"]  = preds_c["unique_id"].astype(str)
    val_df_c["ds"]        = pd.to_datetime(val_df_c["ds"])
    preds_c["ds"]         = pd.to_datetime(preds_c["ds"])

    forecast_col = preds_c.columns.difference(["unique_id", "ds"])[0]

    # --- 4b. Debugging before merge ---
    print("\n=== DEBUG: val_df sample ===")
    print(val_df_c.head(), val_df_c.dtypes)

    print("\n=== DEBUG: preds sample ===")
    print(preds_c.head(), preds_c.dtypes)

    print("\n=== DEBUG: Unique IDs ===")
    print("val_df IDs :", val_df_c["unique_id"].unique())
    print("preds IDs  :", preds_c["unique_id"].unique())

    print("\n=== DEBUG: Date ranges ===")
    print("val_df min/max:", val_df_c["ds"].min(), val_df_c["ds"].max())
    print("preds min/max :", preds_c["ds"].min(), preds_c["ds"].max())

    # See if any exact (ID, ds) pairs match before merging
    val_pairs   = set(zip(val_df_c["unique_id"], val_df_c["ds"]))
    pred_pairs  = set(zip(preds_c["unique_id"], preds_c["ds"]))
    intersection = val_pairs & pred_pairs
    print(f"\n=== DEBUG: Overlapping (unique_id, ds) pairs: {len(intersection)} ===")
    for i, pair in enumerate(sorted(intersection)[:10]):  # show first 10
        print(pair)

    print("🔍 Merging predictions with validation set…")
    merged = val_df_c.merge(preds_c, on=["ds", "unique_id"], how="inner")

    # Drop rows missing actual or forecast
    merged = merged.dropna(subset=["y", forecast_col])

    if merged.empty:
        print("⚠️ No overlap at all — returning large MAE penalty")
        return 1e6  # big number so trial is "bad" but not failed

    # Per-ID MAE, skipping IDs with no overlap
    per_id_mae = merged.groupby("unique_id").apply(
        lambda g: mean_absolute_error(g["y"], g[forecast_col])
    )

    if per_id_mae.empty:
        print("⚠️ No IDs with valid overlap — penalty")
        return 1e6

    avg_mae = per_id_mae.mean()
    print(f"📈 Per-ID MAEs:\n{per_id_mae}\n🔹 Average MAE: {avg_mae}")

    return avg_mae

# 2C. CHRONOS Fine-tuning
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import pandas as pd
from sklearn.metrics import mean_absolute_error

def eval_chronos(trial):
    global train_df, val_df, versions, horizon  # tell Python to use the globals
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

TARGET_MAE = 0.5  # your desired floor for MAE

def stop_when_mae_reached(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    if study.best_value is not None and study.best_value <= TARGET_MAE:
        print(f"🎯 Target MAE {TARGET_MAE} reached (best={study.best_value:.4f}). Stopping study.")
        study.stop()


def main():
    # --------------------------------------------------------------------------------
    # 1. LOAD & PREPROCESS
    # --------------------------------------------------------------------------------
    global train_df, val_df, versions, horizon  # tell Python to use the globals
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

    print(f"🧮 Training set: {train_df.shape[0]} rows")
    print(f"🧮 Validation set: {val_df.shape[0]} rows")
    
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
    study.optimize(objective, 
                   n_trials=75, 
                   # callbacks=[stop_when_mae_reached], use it to call the fuction "stop_when_mae_reached" 
                   timeout=36000) ## XX trials, HH hours

    print("🏆 Best trial parameters:", study.best_trial.params)
    print("📉 Best MAE:", study.best_value)

if __name__ == "__main__":
    main()