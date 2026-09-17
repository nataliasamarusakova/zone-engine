from __future__ import annotations

from pine_r541_sr import Candle, calculate_event_sr, compute_current_sr, build_confirmed_pivots


def _candles(n: int = 400) -> list[Candle]:
    return [Candle(ts=i * 3_600_000, open=50.0, high=100.0, low=0.0, close=50.0, volume=1.0) for i in range(n)]


def test_pine_pivot_is_emitted_on_confirmation_bar():
    candles = _candles()
    # Distinct high pivot at candle 100; confirmation is bar 110.
    candles[100] = Candle(candles[100].ts, 50, 150, 0, 50, 1)
    ph, pl = build_confirmed_pivots(candles, rb=10)
    assert ph[110] == 150.0
    assert ph[100] is None


def test_sr_cluster_uses_recent_pivot_and_strength_two():
    candles = _candles()
    ph = [None] * len(candles)
    pl = [None] * len(candles)
    # Most recent event = 350 at 120; older event = 330 at 121; within cwidth=10.
    ph[350] = 120.0
    ph[330] = 121.0
    # Isolated event at 300 must not form a strength-2 cluster.
    ph[300] = 150.0
    state = calculate_event_sr(candles, ph, pl, 350, rb=10, prd=284, channel_w=10, strength_sr=2)
    assert state["cwidth"] == 10.0
    assert 120.0 in state["levels"]
    assert 150.0 not in state["levels"]
    assert state["highestph"] == 150.0
    assert state["lowestpl"] == 100.0


def test_current_sr_state_persists_after_last_pivot_event():
    candles = []
    for i in range(400):
        close = 50.0 + i * 0.1
        candles.append(Candle(ts=i * 3_600_000, open=close, high=close + 1.0, low=close - 1.0, close=close, volume=1.0))
    # One clear high pivot at candle 340; confirmation is bar 350.
    base = candles[340]
    candles[340] = Candle(base.ts, base.open, 200.0, base.low, base.close, base.volume)
    result = compute_current_sr(candles, rb=10, prd=284, channel_w=10, strength_sr=2)
    # The last event is at 350 and no later pivot exists; its SR state persists
    # through the remaining bars, matching Pine's persistent sr_levs arrays.
    assert result["event_idx"] == 350
    assert result["highestph"] == 200.0


def test_simultaneous_ph_pl_follows_pine_candidate_and_storage_semantics():
    candles = _candles()
    ph = [None] * len(candles)
    pl = [None] * len(candles)
    ph[350] = 120.0
    pl[350] = 80.0
    ph[330] = 121.0
    pl[330] = 79.0
    state = calculate_event_sr(candles, ph, pl, 350, rb=10, prd=284, channel_w=10, strength_sr=2)
    # The candidate for the latest event uses ph precedence for its range, while
    # the Pine storage has the independent pl assignment overwrite ph.
    assert state["highestph"] == 121.0
    assert state["lowestpl"] == 79.0
    # Both events must be clustered; because pl exists, Pine stores the pl value.
    assert 80.0 in state["levels"]
