import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TCN

class NeuralForecastTCN:
    """NeuralForecast TCN implementation for time series forecasting."""
    
    def __init__(self, params):
        """Initialize with Optuna best parameters."""
        self.input_size = params["nf_input_size"]
        self.future_horizon = 90
    
    def _create_model(self, horizon):
        """Create TCN model with specified horizon."""
        return TCN(
            h=horizon,
            input_size=self.input_size,
        )
    
    def _fit_and_predict_validation(self, train_df, val_df, horizon):
        """Fit on train data and predict validation period."""
        print("📈 Fitting NF-TCN on train split…")
        
        model_nf = self._create_model(horizon)
        nf_val = NeuralForecast(models=[model_nf], freq="D")
        nf_val.fit(train_df)
        
        print("🔮 Generating in-sample forecasts…")
        df = nf_val.predict().reset_index()
        pred_val = (
            df[["unique_id", "ds", "TCN"]]
            .rename(columns={"ds": "timestamp", "TCN": "predicted"})
        )
        
        # Restrict to validation window
        pred_val = pred_val[pred_val["timestamp"].isin(val_df["ds"])]
        
        return pred_val
    
    def _fit_and_predict_future(self, full_df):
        """Fit on full data and predict future period."""
        print("📈 Fitting NF-TCN on full history for next-3-mo…")
        
        model_fut = self._create_model(self.future_horizon)
        nf_fut = NeuralForecast(models=[model_fut], freq="D")
        nf_fut.fit(full_df)
        
        print("🔮 Generating 90-day future forecast…")
        df2 = nf_fut.predict().reset_index()
        future_df = (
            df2[["unique_id", "ds", "TCN"]]
            .rename(columns={"ds": "timestamp", "TCN": "future_pred"})
        )
        
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
            data['train_df'], data['val_df'], data['horizon']
        )
        
        # Future predictions
        future_df = self._fit_and_predict_future(data['full_df'])
        
        return pred_val, future_df