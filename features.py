"""
Feature extraction from L2 orderbook data.

Reads from kraken_orderbook.db and computes per-snapshot features:
    - spread, mid_price, log_return
    - realized volatility (rolling windows)
    - bid/ask depth imbalance
    - order flow toxicity (VPIN-style)
    - book pressure (weighted depth near top)
"""

import sqlite3
import json
import numpy as np
import pandas as pd


def load_orderbook_raw(
    db_path: str,
    pair: str = "XBT_USD",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load raw orderbook BBO by forward-filling bid/ask from delta stream.

    Each row in the DB has either best_bid or best_ask (or both on full
    snapshots). Forward-fill reconstructs continuous BBO series.
    """
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT timestamp_unix, best_bid, best_ask, bids_json, asks_json
        FROM {pair}_orderbook
        ORDER BY id
    """
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql(query, conn)
    conn.close()

    # Forward-fill to reconstruct continuous BBO
    df["best_bid"] = df["best_bid"].ffill()
    df["best_ask"] = df["best_ask"].ffill()
    df.dropna(subset=["best_bid", "best_ask"], inplace=True)

    df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2
    df["spread"] = df["best_ask"] - df["best_bid"]

    df.index = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True)
    df.index.name = "timestamp"
    return df


def extract_orderbook_features(
    db_path: str,
    pair: str = "XBT_USD",
    resample: str = "1s",
    vol_windows: list[int] = [30, 60, 300],
    limit: int | None = None,
) -> pd.DataFrame:
    """Extract feature matrix from raw orderbook snapshots.

    Steps:
        1. Load raw BBO stream (~35M rows)
        2. Resample to `resample` interval (default 1s) to reduce size
        3. Compute returns, volatility, depth features

    Returns:
        DataFrame indexed by timestamp with feature columns.
    """
    print(f"Loading {pair} orderbook from {db_path}...")
    raw = load_orderbook_raw(db_path, pair, limit=limit)
    print(f"  Raw rows: {len(raw):,}")

    # Resample to reduce from millions of ticks to manageable bars
    print(f"  Resampling to {resample}...")
    ohlc = raw["mid_price"].resample(resample).ohlc().dropna()
    spread = raw["spread"].resample(resample).median().reindex(ohlc.index)
    best_bid = raw["best_bid"].resample(resample).last().reindex(ohlc.index)
    best_ask = raw["best_ask"].resample(resample).last().reindex(ohlc.index)

    # Parse depth from the last snapshot in each bar (only rows with book data)
    depth = _resample_depth(raw, resample)

    df = pd.DataFrame(index=ohlc.index)
    df["mid_price"] = ohlc["close"]
    df["best_bid"] = best_bid
    df["best_ask"] = best_ask
    df["spread"] = spread
    df["spread_bps"] = df["spread"] / df["mid_price"] * 10_000

    # Returns
    df["log_return"] = np.log(df["mid_price"] / df["mid_price"].shift(1))

    # Realized volatility at multiple windows
    for w in vol_windows:
        df[f"vol_{w}"] = df["log_return"].rolling(w).std() * np.sqrt(w)

    # Spread volatility
    df["spread_vol"] = df["spread_bps"].rolling(60).std()

    # Depth features
    if depth is not None:
        df["bid_depth"] = depth["bid_depth"].reindex(df.index).ffill()
        df["ask_depth"] = depth["ask_depth"].reindex(df.index).ffill()
        df["imbalance"] = depth["imbalance"].reindex(df.index).ffill()
        df["book_pressure"] = depth["book_pressure"].reindex(df.index).ffill()
    else:
        df["bid_depth"] = np.nan
        df["ask_depth"] = np.nan
        df["imbalance"] = 0.0
        df["book_pressure"] = 0.0

    df.dropna(subset=["log_return", f"vol_{vol_windows[-1]}"], inplace=True)
    print(f"  Feature rows: {len(df):,}")
    return df


def _resample_depth(raw: pd.DataFrame, resample: str) -> pd.DataFrame | None:
    """Compute depth features from book JSON, sampled at resample frequency.

    Only processes rows that have non-trivial book data (>1 level).
    Takes the last such row per resample bar.
    """
    # Filter to rows with multi-level book data (full snapshots have 25 levels)
    # Most deltas have 1-3 levels; look for rows with >=5 bracket pairs on either side
    bid_levels = raw["bids_json"].str.count(r"\[")
    ask_levels = raw["asks_json"].str.count(r"\[")
    has_book = (bid_levels >= 5) | (ask_levels >= 5)
    book_rows = raw.loc[has_book].copy()
    if len(book_rows) == 0:
        return None
    print(f"  Depth rows with >=5 levels: {len(book_rows):,}")

    # Sample: take last row per bar to avoid parsing millions of rows
    sampled_idx = book_rows.index.floor(resample)
    book_rows["bar"] = sampled_idx
    sampled = book_rows.groupby("bar").tail(1)

    bid_depths, ask_depths, imbalances, pressures = [], [], [], []
    for _, row in sampled.iterrows():
        bids = _parse_book_side(row["bids_json"])
        asks = _parse_book_side(row["asks_json"])
        # Filter out zero-volume entries (removals)
        bids = [(p, v) for p, v in bids if v > 0]
        asks = [(p, v) for p, v in asks if v > 0]

        bid_depths.append(sum(v for _, v in bids[:10]))
        ask_depths.append(sum(v for _, v in asks[:10]))
        imbalances.append(_compute_depth_imbalance(bids, asks))
        mid = row.get("mid_price") or (row["best_bid"] + row["best_ask"]) / 2 if row["best_bid"] and row["best_ask"] else 0
        pressures.append(_compute_book_pressure(bids, asks, mid) if mid else 0.0)

    result = pd.DataFrame({
        "bid_depth": bid_depths,
        "ask_depth": ask_depths,
        "imbalance": imbalances,
        "book_pressure": pressures,
    }, index=sampled.index)
    # Floor to resample frequency so reindex aligns with feature bars
    result.index = result.index.floor(resample)
    result = result[~result.index.duplicated(keep="last")]
    return result


def _parse_book_side(json_str: str) -> list[tuple[float, float]]:
    """Parse bids_json or asks_json into [(price, volume), ...]."""
    if not json_str:
        return []
    entries = json.loads(json_str)
    return [(float(e[0]), float(e[1])) for e in entries]


def _compute_depth_imbalance(bids: list[tuple], asks: list[tuple], levels: int = 5) -> float:
    """Bid-ask depth imbalance in top N levels. Range [-1, 1]."""
    bid_vol = sum(v for _, v in bids[:levels])
    ask_vol = sum(v for _, v in asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def _compute_book_pressure(bids: list[tuple], asks: list[tuple], mid: float, bps: float = 10) -> float:
    """Volume-weighted depth within `bps` basis points of mid. Positive = bid heavy."""
    threshold = mid * bps / 10_000
    bid_near = sum(v for p, v in bids if mid - p <= threshold)
    ask_near = sum(v for p, v in asks if p - mid <= threshold)
    total = bid_near + ask_near
    if total == 0:
        return 0.0
    return (bid_near - ask_near) / total


if __name__ == "__main__":
    # Quick test with small sample
    df = extract_orderbook_features(
        db_path="/home/eli/research/data/kraken_orderbook.db",
        pair="XBT_USD",
        limit=500_000,
    )
    print(df.head(10))
    print(f"\nColumns: {list(df.columns)}")
    print(df.describe())
