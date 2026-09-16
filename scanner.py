import asyncio
import time
import logging
from dataclasses import dataclass

import aiohttp
import numpy as np


BYBIT_BASE = "https://api.bybit.com"
BINGX_BASE = "https://open-api.bingx.com"


@dataclass
class Signal:
    exchange: str
    symbol: str
    timeframe: str
    price: float
    current_rsi: float
    previous_rsi: float
    latest_rsi: float
    overbought_seen: bool
    quote_volume: float
    previous_pivot_time: str
    latest_pivot_time: str
    latest_pivot_ts: int
    confirmed: bool


def rsi_wilder(closes, period=14):
    x = np.asarray(closes, dtype=float)

    if len(x) < period + 2:
        return np.full(len(x), np.nan)

    delta = np.diff(x)

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(len(x), np.nan)
    avg_loss = np.full(len(x), np.nan)

    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()

    for i in range(period + 1, len(x)):
        avg_gain[i] = (
            avg_gain[i - 1] * (period - 1) + gains[i - 1]
        ) / period

        avg_loss[i] = (
            avg_loss[i - 1] * (period - 1) + losses[i - 1]
        ) / period

    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)

    out = 100 - (100 / (1 + rs))

    out[(avg_loss == 0) & np.isfinite(avg_gain)] = 100

    return out


def pivot_highs(values, left=3, right=3):
    v = np.asarray(values, dtype=float)
    out = []

    for i in range(left, len(v) - right):
        window = v[i - left:i + right + 1]

        if (
            np.isfinite(v[i])
            and v[i] == np.nanmax(window)
            and np.sum(window == v[i]) == 1
        ):
            out.append(i)

    return out


def find_divergence(
    candles,
    rsi_period,
    pivot_left,
    pivot_right,
    max_pivot_gap,
    min_price_higher_high_pct,
    min_rsi_lower_high,
    overbought_rsi,
):
    if len(candles) < rsi_period + pivot_left + pivot_right + 20:
        return None

    closes = [c[4] for c in candles]
    highs = [c[2] for c in candles]

    rsi = rsi_wilder(closes, rsi_period)

    pivots = pivot_highs(
        highs,
        pivot_left,
        pivot_right,
    )

    if len(pivots) < 2:
        return None

    # Второй максимум должен быть свежим:
    # максимум 3 свечи назад.
    MAX_SIGNAL_AGE = 3

    candidates = []

    # Проверяем несколько последних максимумов.
    recent_pivots = pivots[-8:]

    for b in reversed(recent_pivots):

        signal_age = (len(candles) - 1) - b

        if signal_age > MAX_SIGNAL_AGE:
            continue

        price_b = highs[b]
        rsi_b = rsi[b]

        if not np.isfinite(rsi_b):
            continue

        # Ищем предыдущий максимум.
        for a in reversed(pivots):

            if a >= b:
                continue

            gap = b - a

            if gap > max_pivot_gap:
                break

            price_a = highs[a]
            rsi_a = rsi[a]

            if not np.isfinite(rsi_a):
                continue

            # Цена делает Higher High.
            higher_high = (
                price_b
                > price_a * (
                    1 + min_price_higher_high_pct / 100
                )
            )

            # RSI делает Lower High.
            lower_rsi_high = (
                rsi_b
                <= rsi_a - min_rsi_lower_high
            )

            if not (higher_high and lower_rsi_high):
                continue

            # RSI первого максимума должен быть
            # непосредственно выше уровня overbought.
            overbought_seen = (
                rsi_a > overbought_rsi
                and rsi_b > overbought_rsi
            )

            if not overbought_seen:
                continue

            current_rsi = float(rsi[-1])

            if not np.isfinite(current_rsi):
                continue

            # Подтверждение:
            # текущий RSI уже опустился ниже 70.
            confirmed = current_rsi < overbought_rsi

            if not confirmed:
                continue

            candidates.append(
                {
                    "a": a,
                    "b": b,
                    "age": signal_age,
                    "rsi_diff": float(rsi_a - rsi_b),
                }
            )

            break

    if not candidates:
        return None

    # Сначала самый свежий сигнал.
    # При одинаковой свежести —
    # большая разница RSI.
    candidates.sort(
        key=lambda x: (
            x["age"],
            -x["rsi_diff"],
        )
    )

    best = candidates[0]

    a = best["a"]
    b = best["b"]

    price_a = highs[a]
    price_b = highs[b]

    rsi_a = float(rsi[a])
    rsi_b = float(rsi[b])

    current_rsi = float(rsi[-1])

    latest_ts = int(candles[b][0])

    latest_pivot_time = time.strftime(
        "%Y-%m-%d %H:%M UTC",
        time.gmtime(latest_ts / 1000),
    )

    previous_pivot_time = time.strftime(
        "%Y-%m-%d %H:%M UTC",
        time.gmtime(
            int(candles[a][0]) / 1000
        ),
    )

    quote_volume = (
        float(
            sum(
                c[6]
                for c in candles[-24:]
            )
        )
        if len(candles) >= 24
        else 0.0
    )

    logging.info(
        "BEARISH SIGNAL FOUND | "
        "price %.8f -> %.8f | "
        "RSI %.1f -> %.1f | "
        "current RSI %.1f | "
        "age %d candles",
        price_a,
        price_b,
        rsi_a,
        rsi_b,
        current_rsi,
        best["age"],
    )

    return {
        "pivot_index": b,
        "signal": {
            "current_rsi": current_rsi,
            "previous_rsi": rsi_a,
            "latest_rsi": rsi_b,
            "overbought_seen": True,
            "confirmed": True,
            "latest_pivot_ts": latest_ts,
            "latest_pivot_time": latest_pivot_time,
            "previous_pivot_time": previous_pivot_time,
            "price": float(candles[-1][4]),
            "quote_volume": quote_volume,
        },
    }


class ExchangeScanner:
    def __init__(self, name):
        self.name = name.lower()

    async def _get_json(
        self,
        session,
        url,
        params=None,
        headers=None,
    ):
        for attempt in range(3):
            try:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=20,
                ) as r:

                    data = await r.json()

                    if r.status == 200:
                        return data

                    if r.status in (429, 418):
                        await asyncio.sleep(
                            1.5 * (attempt + 1)
                        )
                        continue

                    raise RuntimeError(
                        f"{self.name} HTTP "
                        f"{r.status}: {data}"
                    )

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ValueError,
            ):
                if attempt == 2:
                    raise

                await asyncio.sleep(
                    1.0 * (attempt + 1)
                )

        raise RuntimeError("request failed")

    async def get_usdt_symbols(
        self,
        session,
        min_volume,
        max_symbols,
    ):
        if self.name == "bybit":

            data = await self._get_json(
                session,
                BYBIT_BASE + "/v5/market/tickers",
                {"category": "linear"},
            )

            rows = data["result"]["list"]

            out = []

            for x in rows:

                sym = x["symbol"]

                if not sym.endswith("USDT"):
                    continue

                try:
                    turnover = float(
                        x.get("turnover24h", 0)
                    )
                except Exception:
                    turnover = 0

                if turnover >= min_volume:
                    out.append(
                        (sym, turnover)
                    )

            out.sort(
                key=lambda z: z[1],
                reverse=True,
            )

            return [
                s
                for s, _ in out[:max_symbols]
            ]

        # BingX perpetual USDT
        data = await self._get_json(
            session,
            BINGX_BASE
            + "/openApi/swap/v2/quote/ticker",
        )

        rows = data.get("data", [])

        out = []

        for x in rows:

            sym = x.get("symbol", "")

            if not sym.endswith("-USDT"):
                continue

            try:
                volume = float(
                    x.get("volume", 0)
                )

                last = float(
                    x.get("lastPrice", 0)
                )

                turnover = volume * last

            except Exception:
                turnover = 0

            if turnover >= min_volume:
                out.append(
                    (sym, turnover)
                )

        out.sort(
            key=lambda z: z[1],
            reverse=True,
        )

        return [
            s
            for s, _ in out[:max_symbols]
        ]

    async def get_klines(
        self,
        session,
        symbol,
        timeframe,
        limit=220,
    ):
        if self.name == "bybit":

            interval = {
                "15m": "15",
                "1h": "60",
                "4h": "240",
                "1d": "D",
            }[timeframe]

            data = await self._get_json(
                session,
                BYBIT_BASE
                + "/v5/market/kline",
                {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit,
                },
            )

            rows = data["result"]["list"]

            rows = list(
                reversed(rows)
            )

            return [
                (
                    int(x[0]),
                    float(x[1]),
                    float(x[2]),
                    float(x[3]),
                    float(x[4]),
                    float(x[5]),
                    float(x[6]),
                )
                for x in rows
            ]

        # BingX
        interval = timeframe

        data = await self._get_json(
            session,
            BINGX_BASE
            + "/openApi/swap/v3/quote/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
        )

        rows = data.get("data", [])

        if not isinstance(rows, list):
            return []

        candles = []

        for x in rows:

            try:
                if isinstance(x, dict):

                    ts = int(
                        x.get(
                            "time",
                            x.get(
                                "timestamp",
                                0,
                            ),
                        )
                    )

                    open_price = float(
                        x["open"]
                    )

                    high = float(
                        x["high"]
                    )

                    low = float(
                        x["low"]
                    )

                    close = float(
                        x["close"]
                    )

                    volume = float(
                        x.get("volume", 0)
                    )

                    quote_volume = float(
                        x.get(
                            "quoteVolume",
                            x.get(
                                "quoteAssetVolume",
                                volume * close,
                            ),
                        )
                    )

                elif (
                    isinstance(
                        x,
                        (list, tuple),
                    )
                    and len(x) >= 6
                ):

                    ts = int(x[0])
                    open_price = float(x[1])
                    high = float(x[2])
                    low = float(x[3])
                    close = float(x[4])
                    volume = float(x[5])

                    quote_volume = (
                        float(x[7])
                        if len(x) > 7
                        else volume * close
                    )

                else:
                    continue

                if ts <= 0:
                    continue

                candles.append(
                    (
                        ts,
                        open_price,
                        high,
                        low,
                        close,
                        volume,
                        quote_volume,
                    )
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        candles.sort(
            key=lambda x: x[0]
        )

        return candles

    async def scan_symbols(
        self,
        session,
        symbols,
        timeframe,
        **kwargs,
    ):
        sem = asyncio.Semaphore(
            8
            if self.name == "bybit"
            else 5
        )

        async def one(symbol):

            async with sem:

                try:
                    candles = await self.get_klines(
                        session,
                        symbol,
                        timeframe,
                    )

                    result = find_divergence(
                        candles,
                        **kwargs,
                    )

                    if not result:
                        return None

                    s = result["signal"]

                    return Signal(
                        exchange=self.name,
                        symbol=symbol,
                        timeframe=timeframe,
                        price=s["price"],
                        current_rsi=s["current_rsi"],
                        previous_rsi=s["previous_rsi"],
                        latest_rsi=s["latest_rsi"],
                        overbought_seen=s["overbought_seen"],
                        quote_volume=s["quote_volume"],
                        previous_pivot_time=s[
                            "previous_pivot_time"
                        ],
                        latest_pivot_time=s[
                            "latest_pivot_time"
                        ],
                        latest_pivot_ts=s[
                            "latest_pivot_ts"
                        ],
                        confirmed=s["confirmed"],
                    )

                except Exception as e:

                    logging.exception(
                        "ERROR scanning %s %s %s: %s",
                        self.name,
                        symbol,
                        timeframe,
                        e,
                    )

                    return None

        results = await asyncio.gather(
            *(
                one(s)
                for s in symbols
            )
        )

        return [
            x
            for x in results
            if x is not None
        ]
