"""
Spread optimization conditioned on HMM volatility regime.

Widen spreads in high-vol regimes (adverse selection risk),
tighten in low-vol regimes (capture more flow).
"""

import numpy as np
import pandas as pd
from hmm_model import VolatilityHMM


class SpreadOptimizer:
    """Compute optimal bid-ask spreads given HMM regime state."""

    def __init__(self, hmm: VolatilityHMM, gamma: float = 0.1):
        """
        Args:
            hmm: Fitted VolatilityHMM model
            gamma: Risk aversion parameter for inventory penalty
        """
        self.hmm = hmm
        self.gamma = gamma
        self._regime_base_spreads: np.ndarray | None = None

    def compute_optimal_spreads(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute optimal spread for each timestep.

        Optimal half-spread per regime:
            s_k = c * sigma_k + gamma * sigma_k^2

        Where sigma_k is the regime's characteristic volatility,
        and c is a constant scaling adverse selection cost.
        """
        probs = self.hmm.predict_proba(features)
        regimes = np.argmax(probs, axis=1)

        # Compute per-regime base spread in price terms
        # vol_30 is log-return std, so multiply by mid to get price volatility
        regime_vols = self._estimate_regime_vols(features, regimes)
        avg_mid = features["mid_price"].median()
        self._regime_base_spreads = np.array([
            self._optimal_half_spread(v, avg_mid) for v in regime_vols
        ])

        # Probability-weighted spread: blend across regime probabilities
        weighted_spread = probs @ self._regime_base_spreads

        result = pd.DataFrame(index=features.index)
        result["regime"] = regimes
        result["regime_prob"] = probs[np.arange(len(regimes)), regimes]
        result["optimal_spread"] = weighted_spread * 2  # full spread (both sides)
        result["market_spread"] = features["spread"].values
        result["spread_ratio"] = result["optimal_spread"] / result["market_spread"].clip(lower=1e-10)

        # Include next-regime prediction
        result["next_regime_high_prob"] = np.array([
            self.hmm.predict_next_regime(p)[self.hmm.n_regimes - 1]
            for p in probs
        ])

        for i in range(self.hmm.n_regimes):
            label = self.hmm.regime_labels.get(i, f"regime_{i}")
            print(f"  {label}: base_spread={self._regime_base_spreads[i]:.2f}, "
                  f"vol={regime_vols[i]:.6f}, count={( regimes == i).sum()}")

        return result

    def _estimate_regime_vols(self, features: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Estimate realized volatility per regime from actual data."""
        vols = np.zeros(self.hmm.n_regimes)
        for k in range(self.hmm.n_regimes):
            mask = regimes == k
            if mask.sum() > 0:
                vols[k] = features.loc[mask, "vol_30"].abs().median()
        return vols

    def _optimal_half_spread(self, regime_vol: float, mid_price: float, c: float = 1.0) -> float:
        """Half-spread in price terms.

        sigma_price = regime_vol * mid_price (convert log-return vol to price vol)
        half_spread = c * sigma_price
        """
        sigma_price = regime_vol * mid_price
        return c * sigma_price

    def _fill_probability(self, our_spread: float, market_spread: float) -> float:
        """Probability of fill given our spread vs market spread."""
        if market_spread <= 0:
            return 0.0
        ratio = our_spread / market_spread
        return 1.0 / (1.0 + np.exp(2 * (ratio - 1)))


if __name__ == "__main__":
    from features import extract_orderbook_features

    features = extract_orderbook_features(
        db_path="/home/eli/research/data/kraken_orderbook.db",
        pair="XBT_USD",
        limit=500_000,
    )

    hmm = VolatilityHMM(n_regimes=3)
    hmm.fit(features)

    optimizer = SpreadOptimizer(hmm)
    spreads = optimizer.compute_optimal_spreads(features)
    print(f"\nSpread stats:")
    print(spreads[["regime", "optimal_spread", "market_spread", "spread_ratio"]].describe())
