import pandas as pd
import pandas_ta as ta


def load_and_preprocess_data(csv_path: str):
    """
    Loads BTCUSDT data from CSV and preprocesses with RELATIVE technical features.

    CSV expected columns: [Gmt time, Open, High, Low, Close, Volume]
    Returns (DataFrame, list[str]) — df with all columns, feature_cols for agent.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["Gmt time"],
        dayfirst=True,
    )

    # Strip any trailing spaces in headers
    df.columns = df.columns.str.strip()

    # Datetime index
    df = df.set_index("Gmt time")
    df.sort_index(inplace=True)

    # Ensure numeric OHLCV
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Technical indicators ----
    # RSI and ATR (scale-invariant)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Moving averages
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)

    # MA slopes (first difference)
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    # Distance of price from each MA (relative level)
    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    # MA divergence: MA20 vs MA50
    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # ---- Market regime features ----
    # Volatility regime: rolling ATR percentile (100-bar window)
    df["atr_percentile_100"] = (
        df["atr_14"]
        .rolling(window=100, min_periods=20)
        .rank(pct=True)
    )

    # Trend strength: ADX(14)
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["adx_14"] = adx_df["ADX_14"]

    # Simplified trend direction: +1 bullish (MA20 > MA50), -1 bearish
    df["trend_direction"] = 0
    df.loc[df["ma_20"] > df["ma_50"], "trend_direction"] = 1
    df.loc[df["ma_20"] < df["ma_50"], "trend_direction"] = -1

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (no raw price levels / raw MAs)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20_slope",
        "ma_50_slope",
        "close_ma20_diff",
        "close_ma50_diff",
        "ma_spread",
        "ma_spread_slope",
        "atr_percentile_100",
        "adx_14",
        "trend_direction",
    ]

    return df, feature_cols
