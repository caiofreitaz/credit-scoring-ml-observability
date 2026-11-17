"""Deterministic feature engineering used inside the persisted model pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMTS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMTS = [f"PAY_AMT{i}" for i in range(1, 7)]
MONETARY = ["LIMIT_BAL", *BILL_AMTS, *PAY_AMTS]
ENGINEERED = ["utilization_last", "utilization_avg", "payment_ratio_last", "max_delay", "months_delayed"]


def signed_log1p(values: pd.Series | np.ndarray) -> np.ndarray:
    """Log-compress monetary values keeping the sign (bills can be negative)."""
    arr = np.asarray(values, dtype="float64")
    return np.sign(arr) * np.log1p(np.abs(arr))


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds utilization/payment-behaviour features and log-compresses monetary columns."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CreditFeatureEngineer:  # noqa: N803
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        frame = X.copy()
        limit = frame["LIMIT_BAL"].astype("float64").clip(lower=1.0)
        bills = frame[BILL_AMTS].astype("float64")
        frame["utilization_last"] = (frame["BILL_AMT1"] / limit).clip(-1.0, 5.0)
        frame["utilization_avg"] = (bills.mean(axis=1) / limit).clip(-1.0, 5.0)
        frame["payment_ratio_last"] = (frame["PAY_AMT2"] / frame["BILL_AMT2"].astype("float64").clip(lower=1.0)).clip(
            0.0, 5.0
        )
        frame["max_delay"] = frame[PAY_STATUS].max(axis=1).astype("float64")
        frame["months_delayed"] = (frame[PAY_STATUS] > 0).sum(axis=1).astype("float64")
        for column in MONETARY:
            frame[column] = signed_log1p(frame[column])
        return frame

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(base + ENGINEERED, dtype=object)
