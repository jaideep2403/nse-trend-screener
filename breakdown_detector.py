"""Universal breakdown detector — the 7-signal reversal check, available to EVERY tab.

WHY THIS EXISTS. The owner reported the app saved him from large drawdowns in J&KBANK
and CPPLUS. Measuring what actually fired:

    CPPLUS   peak 2026-07-16 -> −14.2%.  Detector read 2/7 AT THE PEAK
             (RSI divergence + MACD bearish), i.e. before any damage.
    J&KBANK  peak 2026-07-10 -> −20.2%.  Detector read 0/7 at the peak and only
             1/7 at −10.4%. It warned LATE.

That capability lived in `portfolio.py` alone (which is gitignored, local-only) and was
surfaced on exactly one tab. The other 25 scanners listed hundreds of names — Momentum
801, Trending 420, Breakout 339 — with no breakdown warning at all.

This module re-implements the checks STANDALONE (no portfolio.py import, so any scanner
can use it) and adds nothing beyond what was already there: same seven signals, same
thresholds, same labels. Deliberately a faithful port, not an "improvement" — the
behaviour that helped is the behaviour being spread. Whether it EARNS its place is a
separate question answered by `validate()`, which measures whether a high score actually
precedes drawdown rather than assuming it does.

Signals (each +1):
  1 Broke MA20 from above      2 Lower high          3 ADX falling
  4 RSI bearish divergence     5 Distribution volume 6 MACD bearish
  7 Underperforming Nifty
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SCORE = 7
LABELS = (("CRITICAL", 4), ("WARNING", 3), ("EARLY", 2), ("CLEAR", 0))


def label_for(score: int) -> str:
    for name, cut in LABELS:
        if score >= cut:
            return name
    return "CLEAR"


# ── Indicator helpers (self-contained) ───────────────────────────────────────
def _rsi(c: pd.Series, period: int = 14) -> pd.Series | None:
    if len(c) < period + 1:
        return None
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(c: pd.Series) -> pd.Series | None:
    if len(c) < 35:
        return None
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def _swing_highs(c: pd.Series, lookback: int = 60, window: int = 5) -> list[float]:
    s = c.iloc[-lookback:] if len(c) > lookback else c
    out = []
    v = s.values
    for i in range(window, len(v) - window):
        if v[i] == max(v[i - window:i + window + 1]):
            out.append(float(v[i]))
    return out


def adx(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period * 2 + 5 or not {"High", "Low", "Close"} <= set(df.columns):
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    a = dx.ewm(alpha=1 / period, adjust=False).mean()
    v = a.iloc[-1]
    return float(v) if np.isfinite(v) else None


# ── The detector ─────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, nifty: pd.Series | None = None,
             upto: int | None = None) -> dict:
    """Score 0-7 at bar `upto` (default: the last bar). Point-in-time: only bars
    <= upto are ever read, so this is safe to walk through history."""
    if df is None or len(df) < 30:
        return {"score": 0, "max": MAX_SCORE, "reasons": [], "label": "—"}
    if upto is not None:
        df = df.iloc[:upto + 1]
    c = df["Close"].astype(float).dropna()
    if len(c) < 30:
        return {"score": 0, "max": MAX_SCORE, "reasons": [], "label": "—"}
    v = df["Volume"].astype(float).dropna() if "Volume" in df.columns else None

    score, reasons = 0, []
    price = float(c.iloc[-1])
    ma20s = c.rolling(20).mean()
    ma20 = float(ma20s.iloc[-1]) if len(c) >= 20 and np.isfinite(ma20s.iloc[-1]) else None

    # 1. Broke MA20 from above
    if ma20 and len(c) >= 11:
        if price < ma20 and bool((c.iloc[-10:-1] > ma20s.iloc[-10:-1]).any()):
            score += 1; reasons.append("Broke MA20")

    # 2. Lower high
    sh = _swing_highs(c, 60, 5)
    if len(sh) >= 2 and sh[-1] < sh[-2] * 0.99:
        score += 1; reasons.append("Lower high")

    # 3. ADX falling
    a_now = adx(df)
    if a_now is not None and len(df) >= 35:
        a_prev = adx(df.iloc[:-5])
        if a_prev and a_now < a_prev - 3:
            score += 1; reasons.append(f"ADX falling ({a_prev:.0f}→{a_now:.0f})")

    # 4. RSI bearish divergence — a HIGHER price high with a LOWER RSI reading
    rsi = _rsi(c)
    if rsi is not None and len(c) >= 60 and len(rsi) >= 60:
        try:
            recent, prior = c.iloc[-40:], c.iloc[-60:-40]
            ri, pi = int(recent.values.argmax()), int(prior.values.argmax())
            if float(recent.iloc[ri]) > float(prior.iloc[pi]):
                r_idx, p_idx = len(c) - 40 + ri, len(c) - 60 + pi
                if 0 <= r_idx < len(rsi) and 0 <= p_idx < len(rsi):
                    if float(rsi.iloc[r_idx]) < float(rsi.iloc[p_idx]) - 5:
                        score += 1; reasons.append("RSI divergence")
        except Exception:
            pass

    # 5. Distribution volume
    if v is not None and len(v) >= 10 and len(c) >= 10:
        chg = c.iloc[-10:].pct_change()
        vol = v.iloc[-10:]
        try:
            up_v = float(vol[chg > 0].sum()); dn_v = float(vol[chg < 0].sum())
            if dn_v > up_v * 1.3:
                score += 1; reasons.append("Distribution volume")
        except Exception:
            pass

    # 6. MACD bearish 3 sessions running
    hist = _macd_hist(c)
    if hist is not None and len(hist) >= 3 and bool((hist.iloc[-3:] < 0).all()):
        score += 1; reasons.append("MACD bearish")

    # 7. Underperforming Nifty over 20 days
    if nifty is not None and len(nifty) >= 21 and len(c) >= 21:
        try:
            n = nifty.reindex(c.index).ffill()
            if np.isfinite(n.iloc[-1]) and np.isfinite(n.iloc[-21]) and n.iloc[-21] > 0:
                s_chg = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1) * 100
                n_chg = (float(n.iloc[-1]) / float(n.iloc[-21]) - 1) * 100
                if (s_chg - n_chg) < -3:
                    score += 1; reasons.append("Underperforming Nifty")
        except Exception:
            pass

    return {"score": score, "max": MAX_SCORE, "reasons": reasons,
            "label": label_for(score)}


def annotate(rows: list[dict], stocks: dict, nifty: pd.Series | None = None,
             key: str = "symbol") -> list[dict]:
    """Attach breakdown_score / breakdown_label / breakdown_reasons to scanner rows.

    Never raises and never drops a row — a scanner must keep working even if the
    detector fails on one name.
    """
    for r in rows or []:
        try:
            sym = r.get(key)
            df = stocks.get(sym) if sym else None
            if df is None:
                continue
            d = evaluate(df, nifty)
            r["breakdown_score"] = d["score"]
            r["breakdown_label"] = d["label"]
            r["breakdown_reasons"] = d["reasons"]
        except Exception:
            continue
    return rows
