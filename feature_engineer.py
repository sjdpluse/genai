"""
feature_engineer.py — 80+ Advanced Features
"""
import pandas as pd
import numpy as np
import ta
import logging

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    
    # ─── Trend ─────────────────────────────────────────────
    df["ema_20"] = ta.trend.EMAIndicator(c, 20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(c, 200).ema_indicator()
    df["sma_20"] = ta.trend.SMAIndicator(c, 20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(c, 50).sma_indicator()
    
    macd = ta.trend.MACD(c, 26, 12, 9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    
    ichi = ta.trend.IchimokuIndicator(h, l)
    df["ichi_a"] = ichi.ichimoku_a()
    df["ichi_b"] = ichi.ichimoku_b()
    df["ichi_base"] = ichi.ichimoku_base_line()
    df["ichi_conv"] = ichi.ichimoku_conversion_line()
    
    # ─── Momentum ────────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(c, 14).rsi()
    df["rsi_3"] = ta.momentum.RSIIndicator(c, 3).rsi()
    df["rsi_21"] = ta.momentum.RSIIndicator(c, 21).rsi()
    
    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    df["cci"] = ta.trend.CCIIndicator(h, l, c, 20).cci()
    df["mfi"] = ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index()
    df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r()
    
    df["roc_5"] = ta.momentum.ROCIndicator(c, 5).roc()
    df["roc_10"] = ta.momentum.ROCIndicator(c, 10).roc()
    df["roc_20"] = ta.momentum.ROCIndicator(c, 20).roc()
    
    # ─── Volatility ──────────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()
    
    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df["atr_pct"] = df["atr"] / c
    
    kc = ta.volatility.KeltnerChannel(h, l, c, 20, 2)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["kc_pct"] = (c - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"])
    
    # ─── Volume ──────────────────────────────────────────────
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    obv_mean = obv.rolling(50).mean()
    obv_std = obv.rolling(50).std()
    df["obv_z"] = (obv - obv_mean) / obv_std.replace(0, np.nan)
    
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()
    df["vol_ratio"] = v / v.rolling(20).mean()
    df["vol_std"] = v.rolling(20).std() / v.rolling(20).mean()
    
    # ─── Candlestick Patterns ──────────────────────────────
    df["candle_body"] = (c - df["open"]).abs() / df["atr"]
    df["candle_range"] = (h - l) / df["atr"]
    df["upper_wick"] = (h - pd.concat([c, df["open"]], axis=1).max(axis=1)) / df["atr"]
    df["lower_wick"] = (pd.concat([c, df["open"]], axis=1).min(axis=1) - l) / df["atr"]
    
    prev_open = df["open"].shift(1)
    prev_close = c.shift(1)
    bullish_engulfing = (c > df["open"]) & (prev_close < prev_open) & (df["open"] < prev_close) & (c > prev_open)
    bearish_engulfing = (c < df["open"]) & (prev_close > prev_open) & (df["open"] > prev_close) & (c < prev_open)
    df["bullish_engulfing"] = bullish_engulfing.astype(int)
    df["bearish_engulfing"] = bearish_engulfing.astype(int)
    df["doji"] = (df["candle_body"] < 0.1).astype(int)
    
    # ─── Price vs MAs ────────────────────────────────────────────
    df["price_vs_ema20"] = (c - df["ema_20"]) / df["ema_20"]
    df["price_vs_ema50"] = (c - df["ema_50"]) / df["ema_50"]
    df["price_vs_ema200"] = (c - df["ema_200"]) / df["ema_200"]
    df["ema20_vs_ema50"] = (df["ema_20"] - df["ema_50"]) / df["ema_50"]
    df["ema50_vs_ema200"] = (df["ema_50"] - df["ema_200"]) / df["ema_200"]
    
    df["ema20_slope"] = df["ema_20"].diff(3) / df["ema_20"].shift(3)
    df["ema50_slope"] = df["ema_50"].diff(5) / df["ema_50"].shift(5)
    df["ema200_slope"] = df["ema_200"].diff(10) / df["ema_200"].shift(10)
    
    # ─── Support/Resistance & Structure ──────────────────────
    df["dist_to_high20"] = (h.rolling(20).max() - c) / c
    df["dist_to_low20"] = (c - l.rolling(20).min()) / c
    df["dist_to_high50"] = (h.rolling(50).max() - c) / c
    df["dist_to_low50"] = (c - l.rolling(50).min()) / c
    
    df["hh"] = (h > h.rolling(20).max().shift(1)).astype(int)
    df["ll"] = (l < l.rolling(20).min().shift(1)).astype(int)
    df["hl"] = (l > l.rolling(20).min().shift(1)).astype(int)
    df["lh"] = (h < h.rolling(20).max().shift(1)).astype(int)
    
    # ─── Ichimoku & VWAP ───────────────────────────────────
    df["price_vs_ichi_a"] = (c - df["ichi_a"]) / c
    df["price_vs_ichi_b"] = (c - df["ichi_b"]) / c
    df["ichi_cloud_diff"] = (df["ichi_a"] - df["ichi_b"]) / c
    df["price_vs_vwap"] = (c - df["vwap"]) / df["vwap"]
    
    # ─── Divergence Detection ────────────────────────────────
    price_hh = c > c.rolling(10).max().shift(1)
    rsi_lh = df["rsi"] < df["rsi"].rolling(10).max().shift(1)
    df["rsi_bear_div"] = (price_hh & rsi_lh).astype(int)
    
    price_ll = c < c.rolling(10).min().shift(1)
    rsi_hl = df["rsi"] > df["rsi"].rolling(10).min().shift(1)
    df["rsi_bull_div"] = (price_ll & rsi_hl).astype(int)
    
    macd_hh = df["macd_hist"] > df["macd_hist"].rolling(10).max().shift(1)
    macd_lh = df["macd_hist"] < df["macd_hist"].rolling(10).max().shift(1)
    df["macd_bear_div"] = (price_hh & macd_lh).astype(int)
    df["macd_bull_div"] = (price_ll & macd_hh).astype(int)
    
    # ─── Volatility Regime ───────────────────────────────────
    df["vol_regime"] = df["atr_pct"].rolling(50).mean()
    df["is_high_vol"] = (df["atr_pct"] > df["vol_regime"] * 1.5).astype(int)
    
    # ─── Time Features ───────────────────────────────────────
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    return df


def build_feature_matrix(df_1h: pd.DataFrame,
                         df_4h: pd.DataFrame | None = None,
                         df_1d: pd.DataFrame | None = None,
                         funding_df: pd.DataFrame | None = None,
                         macro: dict | None = None) -> pd.DataFrame:
    if "rsi" not in df_1h.columns:
        df_1h = compute_indicators(df_1h)
    else:
        df_1h = df_1h.copy()
    
    feature_cols = [
        "rsi", "rsi_3", "rsi_21", "stoch_k", "stoch_d", "cci", "mfi", "williams_r",
        "macd_hist", "adx", "adx_pos", "adx_neg",
        "bb_pct", "bb_width", "atr_pct", "kc_pct",
        "obv_z", "vol_ratio", "vol_std",
        "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
        "ema20_vs_ema50", "ema50_vs_ema200", "ema20_slope", "ema50_slope", "ema200_slope",
        "candle_body", "candle_range", "upper_wick", "lower_wick",
        "dist_to_high20", "dist_to_low20", "dist_to_high50", "dist_to_low50",
        "roc_5", "roc_10", "roc_20",
        "price_vs_ichi_a", "price_vs_ichi_b", "ichi_cloud_diff", "price_vs_vwap",
        "bullish_engulfing", "bearish_engulfing", "doji",
        "hh", "ll", "hl", "lh",
        "rsi_bear_div", "rsi_bull_div", "macd_bear_div", "macd_bull_div",
        "is_high_vol", "hour", "day_of_week", "is_weekend",
    ]
    
    # 4H with shift(1) — anti-lookahead
    if df_4h is not None:
        if "rsi" not in df_4h.columns:
            df_4h = compute_indicators(df_4h)
        else:
            df_4h = df_4h.copy()
        
        cols_4h = ["rsi", "adx", "macd_hist", "ema20_vs_ema50", "bb_pct",
                   "atr_pct", "price_vs_ema200", "obv_z", "vol_ratio",
                   "price_vs_ema20", "price_vs_ema50", "ema20_slope"]
        df_4h_shifted = df_4h[cols_4h].shift(1)
        df_4h_resampled = df_4h_shifted.reindex(df_1h.index, method="ffill")
        df_4h_resampled.columns = [f"4h_{c}" for c in cols_4h]
        df_1h = pd.concat([df_1h, df_4h_resampled], axis=1)
        feature_cols += [c for c in df_1h.columns if c.startswith("4h_")]
    
    # 1D with shift(1)
    if df_1d is not None:
        if "rsi" not in df_1d.columns:
            df_1d = compute_indicators(df_1d)
        else:
            df_1d = df_1d.copy()
        
        cols_1d = ["rsi", "adx", "macd_hist", "ema20_vs_ema50", "bb_pct",
                   "atr_pct", "price_vs_ema200", "vol_ratio"]
        df_1d_shifted = df_1d[cols_1d].shift(1)
        df_1d_resampled = df_1d_shifted.reindex(df_1h.index, method="ffill")
        df_1d_resampled.columns = [f"1d_{c}" for c in cols_1d]
        df_1h = pd.concat([df_1h, df_1d_resampled], axis=1)
        feature_cols += [c for c in df_1h.columns if c.startswith("1d_")]
    
    # Funding rate
    if funding_df is not None and not funding_df.empty:
        funding_resampled = funding_df.reindex(df_1h.index, method="ffill")
        df_1h["funding_rate"] = funding_resampled["funding_rate"]
        df_1h["funding_ema"] = df_1h["funding_rate"].ewm(span=8).mean()
        feature_cols += ["funding_rate", "funding_ema"]
    else:
        df_1h["funding_rate"] = 0
        df_1h["funding_ema"] = 0
        feature_cols += ["funding_rate", "funding_ema"]
    
    # Macro
    if macro:
        df_1h["dxy_change"] = macro.get("dxy_change", 0)
        df_1h["vix_level"] = macro.get("vix_level", 20)
        df_1h["btc_dominance"] = macro.get("btc_dominance", 50)
        feature_cols += ["dxy_change", "vix_level", "btc_dominance"]
    else:
        df_1h["dxy_change"] = 0
        df_1h["vix_level"] = 20
        df_1h["btc_dominance"] = 50
        feature_cols += ["dxy_change", "vix_level", "btc_dominance"]
    
    for c in feature_cols:
        if c not in df_1h.columns:
            df_1h[c] = np.nan
    
    return df_1h[feature_cols].copy()


def create_labels(df_1h: pd.DataFrame,
                  forward_candles: int = 12,
                  long_threshold: float = 0.018,
                  short_threshold: float = -0.018,
                  time_barrier: int = 24,
                  vol_barrier_mult: float = 2.0) -> pd.Series:
    """
    Triple Barrier Method — برچسب‌گذاری واقعی‌تر
    """
    c = df_1h["close"]
    atr = df_1h["atr"]
    
    future_prices = c.shift(-forward_candles)
    future_return = future_prices / c - 1
    
    labels = pd.Series(0.0, index=df_1h.index, name="label")
    labels[future_return >= long_threshold] = 1.0
    labels[future_return <= short_threshold] = -1.0
    
    # بررسی Vol Barrier (SL) قبل از رسیدن به هدف
    for i in range(len(df_1h) - time_barrier):
        if pd.isna(labels.iloc[i]) or labels.iloc[i] == 0:
            continue
        window = c.iloc[i+1:i+time_barrier+1]
        if labels.iloc[i] == 1.0:
            sl = c.iloc[i] - atr.iloc[i] * vol_barrier_mult
            if (window <= sl).any():
                labels.iloc[i] = -1.0
        elif labels.iloc[i] == -1.0:
            sl = c.iloc[i] + atr.iloc[i] * vol_barrier_mult
            if (window >= sl).any():
                labels.iloc[i] = 1.0
    
    labels[future_return.isna()] = np.nan
    return labels
