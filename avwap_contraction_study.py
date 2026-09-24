"""
Does "contraction above AVWAP" actually work? A point-in-time base-rate study.

The claim (X, @prsablue, OPTIEMUS +40% in 2 days, 2026-09-22/23): price reclaims the
AVWAP anchored at the prior swing high, then CONTRACTS just above it while volume DRIES UP —
"the explosive move often starts quietly". One cherry-picked winner proves nothing, and this
repo's Monster DNA study (464 events) found tightness + volume dry-up did NOT separate winners
(volume BUILD did). So every ingredient is measured separately against the market base rate.

Definitions (all computed at bar t with data <= t only):
  anchor a   = bar of the highest High in [t-120, t-10]   (the swing high the base formed under)
  AVWAP(s)   = sum(TP*V)/sum(V) from a..s, TP = (H+L+C)/3
  PULLBACK   = the post-anchor low sits >= 12% below the anchor high (a real base, not a flag)
  A (reclaim & hold) = the last cross from below->above AVWAP happened 3..15 bars ago, every
                close since is >= 99% of AVWAP, and close[t] is 0..8% above AVWAP (not extended)
  C (contraction)    = last-4-bar range <= 10% of close AND 4-bar avg true range < 0.9 x ATR20
  D (dry-up)         = last-4-bar avg volume < 0.8 x the 50-bar avg volume before the reclaim
  S (surge reclaim)  = reclaim-day volume >= 2 x its prior 50-bar average
Universe: every symbol in the per-symbol store (incl. delisted → survivorship-free), point-in-time
price >= Rs50 and 20d avg turnover >= Rs2cr, ETFs excluded. Entry = NEXT day's open (the signal
is only known after the close). Net of round-trip costs. One event per name per 15 bars.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

os.environ.setdefault("BHAV_DIR", os.path.expanduser("~/.ascent_cache/nse_bhav_days"))
import shared_universe as su
from nse_stocks import is_etf
from costs import round_trip_cost_pct

LOOK_HI, MIN_GAP = 120, 10
H_FWD = (5, 10, 21)
COOLDOWN = 15

GROUPS = {
    "A":         lambda f: f["A"],
    "A+C":       lambda f: f["A"] and f["C"],
    "A+C+D":     lambda f: f["A"] and f["C"] and f["D"],
    "A+C+S":     lambda f: f["A"] and f["C"] and f["S"],
    "A+C+S+D":   lambda f: f["A"] and f["C"] and f["S"] and f["D"],
    "A+S":       lambda f: f["A"] and f["S"],
    "C+D noAVWAP": lambda f: (not f["A"]) and f["C"] and f["Dg"],
    "A+Cl":      lambda f: f["A"] and f["Cl"],
    "A+Cl+Df":   lambda f: f["A"] and f["Cl"] and f["Df"],
    "A+S+Cl+Df": lambda f: f["A"] and f["S"] and f["Cl"] and f["Df"],
}


def _events(sym: str) -> tuple[list, list]:
    df = su.load_symbol_history(sym, 2400)
    if df is None or len(df) < LOOK_HI + 60:
        return [], []
    o, h, l, c, v = (df[k].astype(float).values for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c)
    idx = df.index
    tp = (h + l + c) / 3.0
    cpv = np.concatenate([[0.0], np.cumsum(tp * v)])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    turn20 = pd.Series(c * v).rolling(20).mean().values / 1e7
    vol50 = pd.Series(v).rolling(50).mean().values
    tr = np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
    tr[0] = h[0] - l[0]
    atr20 = pd.Series(tr).rolling(20).mean().values
    ma200 = pd.Series(c).rolling(200).mean().values
    # anchor: argmax High over [t-120, t-10]  → window k covers h[k..k+110], used at t = k+120
    win = swv(h, LOOK_HI - MIN_GAP + 1)
    arg = win.argmax(axis=1)

    ev, base = [], []
    last_ev: dict = {}
    for t in range(LOOK_HI, n):
        if c[t] < 50 or not (turn20[t] >= 2.0):
            continue
        has_fwd = t + max(H_FWD) + 1 < n           # recent bars: conditions yes, returns no
        if has_fwd:
            cost = round_trip_cost_pct(turn20[t]) or 0.3
            entry = o[t + 1]
            if entry <= 0:
                continue
            fwd = {hh: (c[t + hh] / entry - 1) * 100 - cost for hh in H_FWD}
            mx21 = (h[t + 1:t + 22].max() / entry - 1) * 100
            dd21 = (l[t + 1:t + 22].min() / entry - 1) * 100
        else:
            fwd = {hh: np.nan for hh in H_FWD}; mx21 = dd21 = np.nan
        # leader context (point-in-time): above the 200-DMA and up >= 20% over ~6 months
        lead = bool(t >= 200 and c[t] > ma200[t] and c[t] / c[t - 126] - 1 >= 0.20)
        rec = {"sym": sym, "date": idx[t], "fwd5": fwd[5], "fwd10": fwd[10], "fwd21": fwd[21],
               "mx21": mx21, "dd21": dd21, "lead": lead}
        if has_fwd and t % 5 == 0:
            base.append(rec)                       # base rate: every 5th eligible bar

        # contraction + generic dry-up are location-free (also feed the no-AVWAP control)
        rng4 = (h[t - 3:t + 1].max() - l[t - 3:t + 1].min()) / c[t]
        C = bool(rng4 <= 0.10 and tr[t - 3:t + 1].mean() < 0.9 * atr20[t])
        Dg = bool(vol50[t - 4] and v[t - 3:t + 1].mean() < 0.8 * vol50[t - 4])
        A = S = D = Cl = Df = False
        a = (t - LOOK_HI) + int(arg[t - LOOK_HI])
        hi_a = h[a]
        if l[a:t + 1].min() <= hi_a * 0.88:        # a real pullback from the anchor high
            s0 = max(a + 1, t - 16)
            ss = np.arange(s0, t + 1)
            av = (cpv[ss + 1] - cpv[a]) / (cv[ss + 1] - cv[a])
            cls = c[ss]
            above = cls > av
            crosses = np.flatnonzero(above[1:] & ~above[:-1]) + 1
            if len(crosses):
                r_rel = int(crosses[-1]); r = int(ss[r_rel])
                ago = t - r
                held = bool((cls[r_rel:] >= av[r_rel:] * 0.99).all())
                ext = c[t] / av[-1] - 1
                A = bool(3 <= ago <= 15 and held and 0 <= ext <= 0.08)
                if A:
                    pre = vol50[r - 1] if r - 1 >= 0 else np.nan
                    S = bool(pre and v[r] >= 2 * pre)
                    D = bool(pre and v[t - 3:t + 1].mean() < 0.8 * pre)
                    # "visual" reading of the tweet — measured on the bars AFTER the reclaim:
                    post = slice(r + 1, t + 1)
                    Cl = bool((h[post].max() - l[post].min()) / c[t] <= 0.10)   # stayed in a tight box
                    Df = bool(v[post].mean() < 0.5 * v[r])                      # volume faded vs reclaim day
        flags = {"A": A, "C": C, "D": D, "S": S, "Dg": Dg, "Cl": Cl, "Df": Df}
        # each setup definition fires on the FIRST day its own conditions hold (own cooldown),
        # so a setup that completes a few bars after the reclaim is not swallowed by an earlier one
        for g, pred in GROUPS.items():
            if pred(flags) and t - last_ev.get(g, -10**9) >= COOLDOWN:
                ev.append(dict(rec, group=g, **flags))
                last_ev[g] = t
    return ev, base


def _summ(rows, label):
    if not rows:
        print(f"  {label:<44} n=   0"); return
    f = pd.DataFrame(rows).dropna(subset=["fwd21"])
    if not len(f):
        print(f"  {label:<44} n=   0"); return
    hit10 = (f.mx21 >= 10).mean() * 100; hit20 = (f.mx21 >= 20).mean() * 100
    print(f"  {label:<44} n={len(f):>5}  fwd5 {f.fwd5.mean():+5.2f}  fwd10 {f.fwd10.mean():+5.2f}  "
          f"fwd21 {f.fwd21.mean():+6.2f} (med {f.fwd21.median():+5.2f})  win21 {(f.fwd21>0).mean()*100:4.1f}%  "
          f"hit+10% {hit10:4.1f}%  hit+20% {hit20:4.1f}%  avgDD21 {f.dd21.mean():+5.1f}%")


def main():
    syms = sorted(p.stem for p in su._sym_store_dir().glob("*.pkl") if not p.stem.startswith("_"))
    syms = [s for s in syms if not is_etf(s)]
    print(f"scanning {len(syms)} symbols (survivorship-free: incl. delisted)…", flush=True)
    EV, BASE = [], []
    for i, s in enumerate(syms):
        try:
            e, b = _events(s)
            EV += e; BASE += b
        except Exception as ex:
            print("  skip", s, ex)
        if i % 500 == 0:
            print(f"  {i}/{len(syms)} · events so far {len(EV)}", flush=True)
    ev = pd.DataFrame(EV)
    base = pd.DataFrame(BASE)
    cut = base.date.quantile(0.6)
    print(f"\nperiod {base.date.min().date()} → {base.date.max().date()} · IS/OOS cut {cut.date()}")

    labels = {
        "A":           "A  reclaim & hold above AVWAP",
        "A+C":         "A+C  + contraction",
        "A+C+D":       "A+C+D  + dry-up   (the tweet's setup)",
        "A+C+S":       "A+C+S  + surge reclaim, no dry-up req.",
        "A+C+S+D":     "A+C+S+D  surge reclaim then dry-up",
        "A+S":         "A+S  surge reclaim & hold (no contraction)",
        "C+D noAVWAP": "C+D  WITHOUT AVWAP (contraction anywhere)",
        "A+Cl":        "A+Cl  tight box after reclaim (visual)",
        "A+Cl+Df":     "A+Cl+Df  tight box + volume fade (visual)",
        "A+S+Cl+Df":   "A+S+Cl+Df  SURGE reclaim → quiet tight hold",
    }
    groups = [("BASE RATE (all eligible stock-days)", base)] + \
             [(labels[g], ev[ev.group == g]) for g in GROUPS]
    for scope, sel in (("ALL", lambda f: f), ("IN-SAMPLE", lambda f: f[f.date < cut]),
                       ("OUT-OF-SAMPLE", lambda f: f[f.date >= cut])):
        print(f"\n=== {scope} — forward returns net of cost, entry next open ===")
        for label, g in groups:
            _summ(sel(g).to_dict("records") if len(g) else [], label)

    for scope, sel in (("LEADERS ONLY · ALL", lambda f: f[f.lead]),
                       ("LEADERS ONLY · IN-SAMPLE", lambda f: f[f.lead & (f.date < cut)]),
                       ("LEADERS ONLY · OUT-OF-SAMPLE", lambda f: f[f.lead & (f.date >= cut)])):
        print(f"\n=== {scope} — base rate here = leader stock-days ===")
        for label, g in groups:
            if any(k in label for k in ("BASE", "A  reclaim", "A+C+D", "A+S  surge", "A+Cl  tight", "A+S+Cl+Df", "C+D")):
                _summ(sel(g).to_dict("records") if len(g) else [], label)

    # the canonical example should be caught (sanity, not proof)
    opt = ev[(ev.sym == "OPTIEMUS") & (ev.date >= "2026-09-15")]
    print("\nOPTIEMUS sanity:\n", opt[["date", "group", "A", "S", "Cl", "Df", "C", "D"]].to_string(index=False) if len(opt) else "not flagged")
    ev.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".avwap_study_events.pkl"))
    print("\nDone.")


if __name__ == "__main__":
    main()
