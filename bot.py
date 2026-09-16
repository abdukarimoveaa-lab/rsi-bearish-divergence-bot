import os
import asyncio
import logging
import aiohttp

from scanner import ExchangeScanner, Signal


# =========================
# SETTINGS
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGES = ["bybit", "bingx"]

TIMEFRAMES = [
    "15m",
    "1h",
    "4h",
    "1d",
]

RSI_PERIOD = 14

PIVOT_LEFT = int(os.getenv("PIVOT_LEFT", "2"))
PIVOT_RIGHT = int(os.getenv("PIVOT_RIGHT", "2"))

MAX_PIVOT_GAP = int(
    os.getenv("MAX_PIVOT_GAP", "120")
)

MIN_PRICE_HIGHER_HIGH_PCT = float(
    os.getenv(
        "MIN_PRICE_HIGHER_HIGH_PCT",
        "0.2",
    )
)

MIN_RSI_LOWER_HIGH = float(
    os.getenv(
        "MIN_RSI_LOWER_HIGH",
        "0.5",
    )
)

OVERBOUGHT_RSI = float(
    os.getenv(
        "OVERBOUGHT_RSI",
        "70",
    )
)

MIN_VOLUME = float(
    os.getenv(
        "MIN_VOLUME",
        "1000000",
    )
)

MAX_SYMBOLS = int(
    os.getenv(
        "MAX_SYMBOLS",
        "500",
    )
)

SCAN_INTERVAL = int(
    os.getenv(
        "SCAN_INTERVAL",
        "300",
    )
)


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)


# =========================
# TELEGRAM
# =========================

async def telegram_send(session, text):

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    while True:

        async with session.post(
            url,
            json=payload,
            timeout=20,
        ) as r:

            if r.status == 200:
                return

            try:
                body = await r.json()
            except Exception:
                body = {
                    "description": await r.text()
                }

            if r.status == 429:

                retry_after = (
                    body
                    .get("parameters", {})
                    .get("retry_after", 5)
                )

                logging.warning(
                    "Telegram rate limit. "
                    "Waiting %s seconds",
                    retry_after,
                )

                await asyncio.sleep(
                    retry_after + 1
                )

                continue

            raise RuntimeError(
                f"Telegram HTTP "
                f"{r.status}: {body}"
            )


# =========================
# PRICE FORMAT
# =========================

def fmt_price(x):

    if x >= 100:
        return f"{x:.2f}"

    if x >= 1:
        return f"{x:.4f}"

    if x >= 0.01:
        return f"{x:.5f}"

    return (
        f"{x:.8f}"
        .rstrip("0")
        .rstrip(".")
    )


# =========================
# TELEGRAM SIGNAL TEXT
# =========================

def signal_text(s: Signal):

    return (
        "🔴 ПОДТВЕРЖДЁННАЯ BEARISH "
        "DIVERGENCE\n\n"

        f"Биржа: {s.exchange.upper()}\n"
        f"Монета: {s.symbol}\n"
        f"ТФ: {s.timeframe}\n"
        f"Цена: {fmt_price(s.price)}\n\n"

        "Цена: Higher High ✅\n"
        "RSI: Lower High ✅\n\n"

        f"RSI первого максимума: "
        f"{s.previous_rsi:.1f}\n"

        f"RSI второго максимума: "
        f"{s.latest_rsi:.1f}\n"

        f"Разница RSI: "
        f"-{abs(s.previous_rsi - s.latest_rsi):.1f}\n"

        f"RSI сейчас: "
        f"{s.current_rsi:.1f}\n\n"

        f"RSI первого максимума >70: "
        f"{'✅' if s.overbought_seen else '❌'}\n"

        f"RSI сейчас <70: "
        f"{'✅' if s.current_rsi < OVERBOUGHT_RSI else '❌'}\n\n"

        f"24h объём: "
        f"${s.quote_volume:,.0f}\n"

        f"Пивоты: "
        f"{s.previous_pivot_time} → "
        f"{s.latest_pivot_time}\n\n"

        "⚠️ Это технический сигнал, "
        "а не гарантия разворота."
    )


# =========================
# SCAN
# =========================

async def scan_once(state):

    scanners = [
        ExchangeScanner(x)
        for x in EXCHANGES
    ]

    all_signals = []

    async with aiohttp.ClientSession() as session:

        for scanner in scanners:

            try:

                symbols = await scanner.get_usdt_symbols(
                    session,
                    min_volume=MIN_VOLUME,
                    max_symbols=MAX_SYMBOLS,
                )

                logging.info(
                    "%s: %d symbols selected",
                    scanner.name,
                    len(symbols),
                )

                for tf in TIMEFRAMES:

                    signals = await scanner.scan_symbols(
                        session,
                        symbols,
                        tf,
                        rsi_period=RSI_PERIOD,
                        pivot_left=PIVOT_LEFT,
                        pivot_right=PIVOT_RIGHT,
                        max_pivot_gap=MAX_PIVOT_GAP,
                        min_price_higher_high_pct=(
                            MIN_PRICE_HIGHER_HIGH_PCT
                        ),
                        min_rsi_lower_high=(
                            MIN_RSI_LOWER_HIGH
                        ),
                        overbought_rsi=(
                            OVERBOUGHT_RSI
                        ),
                    )

                    all_signals.extend(signals)

                    logging.info(
                        "%s %s: %d bearish signals",
                        scanner.name,
                        tf,
                        len(signals),
                    )

            except Exception:

                logging.exception(
                    "Scan failed for %s",
                    scanner.name,
                )

        # Отправляем только новые сигналы.
        for s in all_signals:

            key = (
                s.exchange,
                s.symbol,
                s.timeframe,
                s.latest_pivot_ts,
            )

            if key in state:
                continue

            await telegram_send(
                session,
                signal_text(s),
            )

            state.add(key)

            # Небольшая пауза между сообщениями
            # для защиты от Telegram rate limit.
            await asyncio.sleep(1.1)


# =========================
# MAIN
# =========================

async def main():

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing"
        )

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing"
        )

    state = set()

    logging.info(
        "Bearish divergence scanner started"
    )

    logging.info(
        "Timeframes: %s",
        ", ".join(TIMEFRAMES),
    )

    while True:

        try:

            await scan_once(state)

        except Exception:

            logging.exception(
                "Scan cycle failed"
            )

        logging.info(
            "Next scan in %d seconds",
            SCAN_INTERVAL,
        )

        await asyncio.sleep(
            SCAN_INTERVAL
        )


if __name__ == "__main__":
    asyncio.run(main())
