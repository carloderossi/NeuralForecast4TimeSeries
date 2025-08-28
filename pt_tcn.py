import pandas as pd
import torch
from torch.utils.data import DataLoader
from tune_opt_forecats import EstateDataset, TCNModel

class PyTorchTCN:
    """PyTorch TCN implementation for time series forecasting."""
    
    def __init__(self, params):
        """Initialize with Optuna best parameters."""
        self.seq_len = params["pt_seq_len"]
        self.lr = params["pt_lr"]
        self.dropout = params["pt_dropout"]
        self.n_layers = params["pt_n_layers"]
        self.channels = [params[f"pt_ch{i}"] for i in range(self.n_layers)]
        self.future_horizon = 180 # 3 months=90 # 6 months=180 #12 months=360
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _validate_data_length(self, train_df, horizon, versions):
        """Validate that series are long enough for the given parameters."""
        series_lengths = {
            uid: len(train_df[train_df.unique_id == uid])
            for uid in versions
        }
        print("🔍 Series lengths:", series_lengths)
        min_len = min(series_lengths.values())
        
        if self.seq_len + horizon > min_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) + horizon ({horizon}) = "
                f"{self.seq_len + horizon} > shortest series length ({min_len}). "
                "Either reduce seq_len or reduce your horizon."
            )
    
    def _create_model(self, horizon):
        """Create TCN model with specified horizon."""
        return TCNModel(
            seq_len=self.seq_len,
            channels=self.channels,
            dropout=self.dropout,
            horizon=horizon
        ).to(self.device)
    
    def _train_model(self, model, dataloader, epochs=5):
        import torch
        """Train the model for specified epochs."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # pick GPU if available, otherwise fall back to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train()
        for epoch in range(epochs):
            for xb, yb, mb in dataloader:   # ✅ use the function parameter here
                xb = xb.unsqueeze(1).to(device)
                yb = yb.to(device)
                mb = mb.to(device)

                preds = model(xb)
                # masked MAE
                loss = (torch.abs(preds - yb) * mb).sum() / mb.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def _generate_predictions(self, model, df, versions, horizon, is_future=False):
        """Generate predictions for given data and versions."""
        rows = []
        model.eval()
        
        with torch.no_grad():
            for uid in versions:
                ser = df[df.unique_id == uid].sort_values("ds")
                x = ser["y"].values[-self.seq_len:]
                
                inp = (torch.tensor(x)
                      .unsqueeze(0).unsqueeze(0)
                      .float().to(self.device))
                out = model(inp).cpu().numpy().ravel()
                
                if is_future:
                    # Generate future dates
                    start = ser["ds"].max() + pd.Timedelta(days=1)
                    dates = [start + pd.Timedelta(days=i) for i in range(horizon)]
                    rows += [(uid, d, float(p)) for d, p in zip(dates, out)]
                else:
                    # Use validation dates
                    val_dates = df[df.unique_id == uid].sort_values("ds")["ds"].values
                    for d, p in zip(val_dates, out):
                        rows.append((uid, pd.to_datetime(d), float(p)))
        
        return rows
    
    def _fit_and_predict_validation(self, train_df, val_df, horizon, versions):
        """Fit on train data and predict validation period."""
        self._validate_data_length(train_df, horizon, versions)
        
        # Create model and dataset
        model = self._create_model(horizon)
        train_ds = EstateDataset(train_df, self.seq_len, horizon)
        
        print("🔍 train_ds length:", len(train_ds))
        if len(train_ds) == 0:
            raise ValueError(
                "Your EstateDataset returned zero samples. "
                "Check that your DataFrame has columns ['ds','y','unique_id'] "
                "and that seq_len+horizon < length of each series."
            )
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        print("📈 Fine-tuning PT-TCN on train split…")
        self._train_model(model, train_loader)
        
        print("🔮 Generating in-sample forecasts…")
        rows = self._generate_predictions(model, train_df, versions, horizon, is_future=False)
        
        pred_val = pd.DataFrame(rows, columns=["unique_id", "timestamp", "predicted"])
        return pred_val
    
    def _fit_and_predict_future(self, full_df, versions):
        """Fit on full data and predict future period."""
        model_fut = self._create_model(self.future_horizon)
        full_ds = DataLoader(
            EstateDataset(full_df, self.seq_len, self.future_horizon),
            batch_size=32, shuffle=True
        )
        
        print(f"📈 Fine-tuning PT-TCN on full history for next {self.future_horizon} days")
        self._train_model(model_fut, full_ds)
        
        print("🔮 Generating 90-day future forecast…")
        rows = self._generate_predictions(
            model_fut, full_df, versions, self.future_horizon, is_future=True
        )
        
        future_df = pd.DataFrame(rows, columns=["unique_id", "timestamp", "future_pred"])
        return future_df
    
    def forecast(self, data):
        """
        Generate both validation and future forecasts.
        
        Args:
            data (dict): Dictionary containing train_df, val_df, full_df, versions, horizon
            
        Returns:
            tuple: (pred_val, future_df) - validation and future predictions
        """
        # Validation predictions
        pred_val = self._fit_and_predict_validation(
            data['train_df'], data['val_df'], data['horizon'], data['versions']
        )
        
        # Future predictions
        future_df = self._fit_and_predict_future(data['full_df'], data['versions'])
        
        return pred_val, future_df