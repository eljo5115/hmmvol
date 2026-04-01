"""
HMMVol — HMM Volatility Regime Detection for Spread Optimization

Pipeline:
    1. Extract features from L2 orderbook data (spread, depth imbalance, volatility)
    2. Fit HMM to detect volatility regimes (low/medium/high)
    3. Optimize spread placement conditioned on current regime
    4. Simulate market making and compare vs fixed-spread baseline
"""

from features import extract_orderbook_features
from hmm_model import VolatilityHMM
from spread_optimizer import SpreadOptimizer
from simulator import Simulator, SimConfig

DB_PATH = "/home/eli/research/data/kraken_orderbook.db"
PAIR = "XBT_USD"
LIMIT = 2_000_000  # rows to load (None for full dataset)


def main():
    # 1. Extract features
    features = extract_orderbook_features(db_path=DB_PATH, pair=PAIR, limit=LIMIT)

    # 2. Fit HMM on first 80%, test on last 20%
    split = int(len(features) * 0.8)
    train = features.iloc[:split]
    test = features.iloc[split:]
    print(f"\nTrain: {len(train):,} bars, Test: {len(test):,} bars")

    hmm = VolatilityHMM(n_regimes=3)
    hmm.fit(train)

    # 3. Compute regime-conditioned optimal spreads on test set
    optimizer = SpreadOptimizer(hmm)
    optimal_spreads = optimizer.compute_optimal_spreads(test)

    # 4. Simulate market making
    sim = Simulator(config=SimConfig(quote_size=0.01, max_inventory=0.1))

    print("\n--- REGIME-CONDITIONED SPREADS ---")
    result = sim.run(test, optimal_spreads)
    print(result.summary())

    # 5. Baseline: fixed spread at 1.5x median market spread
    baseline_spreads = optimal_spreads.copy()
    baseline_spreads["optimal_spread"] = test["spread"].median() * 1.5
    print("\n--- FIXED SPREAD BASELINE ---")
    baseline = sim.run(test, baseline_spreads)
    print(baseline.summary())

    print(f"\nRegime-conditioned Sharpe: {result.sharpe:.2f}")
    print(f"Fixed spread Sharpe:      {baseline.sharpe:.2f}")


if __name__ == "__main__":
    main()
