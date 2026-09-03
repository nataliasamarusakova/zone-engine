from run_once import WATCHLIST_ONLY, WATCHLIST_SYMBOLS


def test_watchlist_is_exact_20_symbols():
    expected = (
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "TAO-USDT",
        "LTC-USDT", "BCH-USDT", "AVAX-USDT", "LINK-USDT", "ETC-USDT",
        "ADA-USDT", "UNI-USDT", "XRP-USDT", "ICP-USDT", "HYPE-USDT",
        "DOGE-USDT", "HBAR-USDT", "ARB-USDT", "POL-USDT", "SUI-USDT",
    )
    assert WATCHLIST_ONLY is False
    assert WATCHLIST_SYMBOLS == expected
    assert len(WATCHLIST_SYMBOLS) == 20
