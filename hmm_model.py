"""
HMM-based volatility regime detection.

Fits a Gaussian HMM to orderbook-derived features to identify
discrete volatility regimes (e.g. low / medium / high).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


# Features used as HMM observations
HMM_OBS_COLS = ["vol_30", "vol_60", "spread_bps", "imbalance"]


class VolatilityHMM:
    """Gaussian HMM for volatility regime detection."""

    def __init__(self, n_regimes: int = 3, covariance_type: str = "diag",
                 obs_cols: list[str] | None = None):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.obs_cols = obs_cols or HMM_OBS_COLS
        self.model: GaussianHMM | None = None
        self.scaler = StandardScaler()
        self._sort_order: np.ndarray | None = None  # maps fitted states → sorted states
        self.regime_labels = {0: "low_vol", 1: "med_vol", 2: "high_vol"}

    def fit(self, features: pd.DataFrame) -> "VolatilityHMM":
        """Fit HMM to feature matrix.

        After fitting, regimes are sorted by mean vol_30
        so regime 0 = lowest volatility, regime N-1 = highest.
        """
        obs = self._prepare_obs(features, fit_scaler=True)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=100,
            tol=1e-4,
            random_state=42,
        )
        self.model.fit(obs)

        # Sort regimes by mean volatility (first obs col = vol_30)
        vol_col_idx = 0  # vol_30 is first in obs_cols
        mean_vols = self.model.means_[:, vol_col_idx]
        self._sort_order = np.argsort(mean_vols)

        # Rearrange model parameters so regime 0 = lowest vol
        self.model.means_ = self.model.means_[self._sort_order].copy()
        # covars_ setter validates shape — use internal attribute directly
        self.model._covars_ = self.model._covars_[self._sort_order].copy()
        self.model.startprob_ = self.model.startprob_[self._sort_order].copy()
        self.model.transmat_ = self.model.transmat_[self._sort_order][:, self._sort_order].copy()

        print(f"HMM fitted: {self.n_regimes} regimes, "
              f"converged={self.model.monitor_.converged}, "
              f"iters={self.model.monitor_.iter}")
        for i in range(self.n_regimes):
            print(f"  Regime {i} ({self.regime_labels.get(i, '?')}): "
                  f"mean_vol={self.model.means_[i, vol_col_idx]:.6f}")

        return self

    def decode(self, features: pd.DataFrame) -> np.ndarray:
        """Return most likely regime sequence (Viterbi)."""
        obs = self._prepare_obs(features)
        return self.model.decode(obs)[1]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return regime probabilities per timestep (forward-backward)."""
        obs = self._prepare_obs(features)
        return self.model.predict_proba(obs)

    def predict_next_regime(self, current_probs: np.ndarray) -> np.ndarray:
        """One-step-ahead regime probability from transition matrix."""
        return current_probs @ self.transition_matrix

    @property
    def transition_matrix(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.transmat_

    @property
    def regime_means(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.means_

    def _prepare_obs(self, features: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Extract and scale observation matrix from features."""
        raw = features[self.obs_cols].values.astype(np.float64)
        raw = np.nan_to_num(raw, nan=0.0)
        if fit_scaler:
            return self.scaler.fit_transform(raw)
        return self.scaler.transform(raw)


if __name__ == "__main__":
    from features import extract_orderbook_features

    features = extract_orderbook_features(
        db_path="/home/eli/research/data/kraken_orderbook.db",
        pair="XBT_USD",
        limit=500_000,
    )

    hmm = VolatilityHMM(n_regimes=3)
    hmm.fit(features)

    regimes = hmm.decode(features)
    probs = hmm.predict_proba(features)

    print(f"\nRegime distribution:")
    for i in range(3):
        pct = (regimes == i).mean() * 100
        print(f"  {hmm.regime_labels[i]}: {pct:.1f}%")

    print(f"\nTransition matrix:")
    print(np.round(hmm.transition_matrix, 3))
