"""
Backtest simulator for spread strategies against historical L2 orderbook data.

Replays orderbook snapshots and simulates a market maker quoting
at the spreads produced by SpreadOptimizer.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Fill:
    timestamp: float
    side: str          # "buy" or "sell"
    price: float
    size: float
    mid_at_fill: float


@dataclass
class SimConfig:
    quote_size: float = 0.01          # BTC per side
    max_inventory: float = 0.1        # max absolute position
    inventory_skew: float = 0.5       # how much to skew quotes toward flat
    maker_rebate_bps: float = 0.0
    mark_to_market: bool = True


@dataclass
class SimState:
    position: float = 0.0
    cash: float = 0.0
    realized_pnl: float = 0.0
    n_fills: int = 0
    n_adverse: int = 0
    fills: list[Fill] = field(default_factory=list)
    pnl_history: list[float] = field(default_factory=list)
    timestamps: list = field(default_factory=list)


class Simulator:
    """Replay orderbook data and simulate market making."""

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()

    def run(
        self,
        features: pd.DataFrame,
        optimal_spreads: pd.DataFrame,
    ) -> "SimResult":
        """Run backtest.

        Args:
            features: DataFrame with best_bid, best_ask, mid_price columns
            optimal_spreads: DataFrame with optimal_spread column (in price terms)
        """
        cfg = self.config
        state = SimState()

        # Align indices
        idx = features.index.intersection(optimal_spreads.index)
        feat = features.loc[idx]
        spreads = optimal_spreads.loc[idx, "optimal_spread"].values
        mids = feat["mid_price"].values
        bids = feat["best_bid"].values
        asks = feat["best_ask"].values

        for i in range(len(idx) - 1):
            mid = mids[i]
            half_spread = spreads[i] / 2

            if half_spread <= 0 or np.isnan(mid):
                state.pnl_history.append(state.realized_pnl + self._unrealized(state, mid))
                state.timestamps.append(idx[i])
                continue

            # Compute quotes with inventory skew
            our_bid, our_ask = self._compute_quotes(mid, half_spread, state.position)

            # Check fills against NEXT bar's market prices
            next_ask = asks[i + 1]  # market best ask at next tick
            next_bid = bids[i + 1]  # market best bid at next tick
            next_mid = mids[i + 1]

            # Bid fill: market ask drops to our bid (someone sells to us)
            if next_ask <= our_bid and abs(state.position) < cfg.max_inventory:
                fill = Fill(
                    timestamp=idx[i],
                    side="buy",
                    price=our_bid,
                    size=cfg.quote_size,
                    mid_at_fill=mid,
                )
                state.position += cfg.quote_size
                state.cash -= our_bid * cfg.quote_size
                state.n_fills += 1
                if next_mid < our_bid:
                    state.n_adverse += 1
                state.fills.append(fill)

            # Ask fill: market bid rises to our ask (someone buys from us)
            if next_bid >= our_ask and abs(state.position) < cfg.max_inventory:
                fill = Fill(
                    timestamp=idx[i],
                    side="sell",
                    price=our_ask,
                    size=cfg.quote_size,
                    mid_at_fill=mid,
                )
                state.position -= cfg.quote_size
                state.cash += our_ask * cfg.quote_size
                state.n_fills += 1
                if next_mid > our_ask:
                    state.n_adverse += 1
                state.fills.append(fill)

            # Track P&L
            total_pnl = state.cash + state.position * mid  # mark-to-market
            state.pnl_history.append(total_pnl)
            state.timestamps.append(idx[i])

        # Final mark-to-market
        state.realized_pnl = state.cash + state.position * mids[-1]

        pnl_curve = pd.Series(state.pnl_history, index=state.timestamps, name="pnl")

        max_dd = 0.0
        peak = pnl_curve.iloc[0] if len(pnl_curve) > 0 else 0
        for v in pnl_curve:
            peak = max(peak, v)
            max_dd = min(max_dd, v - peak)

        metrics = {
            "total_pnl": state.realized_pnl,
            "n_fills": state.n_fills,
            "n_adverse": state.n_adverse,
            "adverse_pct": state.n_adverse / max(state.n_fills, 1) * 100,
            "max_inventory": max(abs(f.size) for f in state.fills) if state.fills else 0,
            "max_drawdown": max_dd,
            "final_position": state.position,
            "duration": f"{(idx[-1] - idx[0]).total_seconds() / 3600:.1f}h",
        }

        return SimResult(pnl_curve=pnl_curve, fills=state.fills,
                         final_state=state, metrics=metrics)

    def _compute_quotes(self, mid: float, half_spread: float, position: float) -> tuple[float, float]:
        """Bid/ask with inventory skew."""
        skew = self.config.inventory_skew * position * half_spread
        bid = mid - half_spread - skew
        ask = mid + half_spread - skew
        return bid, ask

    def _unrealized(self, state: SimState, mid: float) -> float:
        return state.position * mid


@dataclass
class SimResult:
    pnl_curve: pd.Series
    fills: list[Fill]
    final_state: SimState
    metrics: dict

    @property
    def sharpe(self) -> float:
        returns = self.pnl_curve.diff().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        # Assume 1-second bars
        periods_per_year = 365.25 * 24 * 3600
        return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

    def summary(self) -> str:
        m = self.metrics
        return "\n".join([
            f"{'='*50}",
            f"SIMULATION RESULTS",
            f"{'='*50}",
            f"Duration:          {m.get('duration', 'N/A')}",
            f"Total P&L:         ${m.get('total_pnl', 0):.2f}",
            f"Total fills:       {m.get('n_fills', 0)}",
            f"Adverse fills:     {m.get('n_adverse', 0)} ({m.get('adverse_pct', 0):.1f}%)",
            f"Final position:    {m.get('final_position', 0):.4f} BTC",
            f"Max drawdown:      ${m.get('max_drawdown', 0):.2f}",
            f"Sharpe:            {self.sharpe:.2f}",
            f"{'='*50}",
        ])
