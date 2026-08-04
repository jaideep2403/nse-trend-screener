/* ── Alpha Engine — Institutional Multi-Factor Composite Scanner ── */
/* LOCAL-ONLY: gitignored — never pushed to GitHub                   */

(function () {
  "use strict";

  // ── State ───────────────────────────────────────────────────────────────────
  let _data        = null;
  let _filtered    = [];
  let _sortCol     = "score";
  let _sortDir     = -1;          // -1 = desc, 1 = asc
  let _tierFilter  = "all";
  let _exitFilter  = false;
  let _foFilter    = "all";       // "all" | "bullish" | "bearish"
  let _searchText  = "";
  let _polling     = null;

  const TIER_ORDER = { BUY: 0, STRONG: 1, MONITOR: 2, AVOID: 3 };

  // ── Tier config ─────────────────────────────────────────────────────────────
  const TIER_CFG = {
    BUY:     { emoji: "🏆", label: "BUY",     color: "var(--green)"  },
    STRONG:  { emoji: "💎", label: "STRONG",   color: "var(--gold)"   },
    MONITOR: { emoji: "👀", label: "MONITOR",  color: "var(--accent2)"},
    AVOID:   { emoji: "⛔", label: "AVOID",    color: "var(--muted)"  },
  };

  const EXIT_CFG = {
    DELIV_DROP:  { label: "Del↓",       title: "Delivery % collapsed — distribution" },
    BULK_SELL:   { label: "InstSell",   title: "Institutional sell in bulk/block deals" },
    OI_SHORT:    { label: "OI:Short",   title: "F&O short buildup — bearish OI signal" },
    OI_UNWIND:   { label: "OI:Unwind",  title: "Long unwinding — smart money exiting" },
    DIST_DAYS:   { label: "4DistDays",  title: "4+ distribution days in last 20 sessions" },
    RS_WEAK:     { label: "RS↓",        title: "Underperforming Nifty by >5% (6M)" },
    MA50_BREAK:  { label: "MA50↓",      title: "Price broke below MA50" },
  };

  // ── Factor bar (mini 0-100 progress bar) ─────────────────────────────────
  function _factorBar(val, max_val, color) {
    const pct = Math.min(100, Math.round((val / max_val) * 100));
    return `<span style="display:inline-flex;align-items:center;gap:4px">
      <span style="font-weight:600;min-width:20px;text-align:right">${val}</span>
      <span style="display:inline-block;width:32px;height:4px;background:var(--border);border-radius:2px;overflow:hidden">
        <span style="display:block;width:${pct}%;height:100%;background:${color};border-radius:2px"></span>
      </span>
    </span>`;
  }

  // ── Colour helpers ──────────────────────────────────────────────────────────
  function _numColour(v, good, bad) {
    if (v === null || v === undefined) return "var(--text2)";
    return v >= good ? "var(--green)" : v <= bad ? "var(--red)" : "var(--gold)";
  }

  function _fmt(v, decimals, suffix) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    return v.toFixed(decimals) + (suffix || "");
  }

  // ── Render exit flags row ───────────────────────────────────────────────────
  function _exitBadges(flags) {
    if (!flags || flags.length === 0) return '<span style="color:var(--green2);font-size:var(--fs-micro)">✓ Clean</span>';
    return flags.map(f => {
      const cfg = EXIT_CFG[f] || { label: f, title: f };
      return `<span class="tag" title="${cfg.title}" style="background:rgba(239,68,68,0.15);color:var(--red2);border:1px solid rgba(239,68,68,0.3);font-size:var(--fs-2xs);cursor:help">${cfg.label}</span>`;
    }).join(" ");
  }

  // ── FO signal badge ─────────────────────────────────────────────────────────
  function _foBadge(sig) {
    if (!sig || sig === "—") return '<span style="color:var(--muted);font-size:var(--fs-micro)">—</span>';
    const map = {
      "Long Buildup":  { cls: "var(--green2)",  icon: "📈" },
      "Short Cover":   { cls: "var(--accent2)", icon: "🔄" },
      "Short Buildup": { cls: "var(--red2)",    icon: "📉" },
      "Long Unwind":   { cls: "var(--red2)",    icon: "⚠️" },
      "Neutral":       { cls: "var(--muted)",   icon: "⟷" },
    };
    const cfg = map[sig] || { cls: "var(--text2)", icon: "" };
    return `<span style="color:${cfg.cls};font-size:var(--fs-micro)">${cfg.icon} ${sig}</span>`;
  }

  // ── Score gradient cell ─────────────────────────────────────────────────────
  function _scoreCell(score, tier) {
    const cfg  = TIER_CFG[tier] || TIER_CFG.MONITOR;
    const pct  = Math.round(score);
    const grad = `conic-gradient(${cfg.color} ${pct * 3.6}deg, var(--border) 0deg)`;
    return `
      <div style="display:inline-flex;flex-direction:column;align-items:center;gap:1px">
        <span style="font-size:var(--fs-lg);font-weight:800;color:${cfg.color};font-family:var(--mono)">${score}</span>
        <span style="font-size:var(--fs-2xs);color:${cfg.color};font-weight:700">${cfg.emoji} ${cfg.label}</span>
      </div>`;
  }

  // ── Main render ─────────────────────────────────────────────────────────────
  window.aeRender = function () {
    if (!_data) return;
    const results = _data.results || [];

    // Filter
    _filtered = results.filter(r => {
      if (_tierFilter === "buy"    && r.tier !== "BUY")    return false;
      if (_tierFilter === "strong" && !["BUY","STRONG"].includes(r.tier)) return false;
      if (_tierFilter === "avoid"  && r.tier === "AVOID")  return false;
      if (_exitFilter && !r.has_exit)  return false;
      if (_foFilter === "bullish" && !["Long Buildup","Short Cover"].includes(r.fo_signal)) return false;
      if (_foFilter === "bearish" && !["Short Buildup","Long Unwind"].includes(r.fo_signal)) return false;
      if (_searchText) {
        const q = _searchText.toLowerCase();
        if (!r.symbol.toLowerCase().includes(q) && !(r.sector||"").toLowerCase().includes(q)) return false;
      }
      return true;
    });

    // Sort
    _filtered.sort((a, b) => {
      let av = a[_sortCol], bv = b[_sortCol];
      if (av === null || av === undefined) av = _sortDir > 0 ? Infinity : -Infinity;
      if (bv === null || bv === undefined) bv = _sortDir > 0 ? Infinity : -Infinity;
      return (av < bv ? -1 : av > bv ? 1 : 0) * _sortDir;
    });

    // Count badge
    document.getElementById("aeCnt").textContent = `${_filtered.length} of ${results.length}`;

    // Tbody
    const tbody = document.getElementById("aeTbody");
    if (_filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="20" style="text-align:center;padding:30px;color:var(--muted)">No stocks match the current filters.</td></tr>`;
      return;
    }

    tbody.innerHTML = _filtered.map((r, i) => {
      const tierCfg  = TIER_CFG[r.tier] || TIER_CFG.MONITOR;
      const rowAlpha = r.has_exit ? "0.85" : "1";
      const rowBg    = r.tier === "BUY"    ? "rgba(16,185,129,0.04)" :
                       r.tier === "STRONG" ? "rgba(245,158,11,0.04)" : "transparent";
      const exitRow  = r.has_exit ? `<tr style="background:rgba(239,68,68,0.06)"><td colspan="20" style="padding:3px 10px;font-size:var(--fs-micro);color:var(--red2)">⚠ Exit Signals: ${_exitBadges(r.exit_flags)}</td></tr>` : "";

      return `
        <tr style="border-bottom:1px solid var(--border);opacity:${rowAlpha};background:${rowBg}"
            data-sym="${r.symbol}">
          <td style="padding:6px 8px;color:var(--muted);font-size:var(--fs-xs)">${i + 1}</td>
          <td style="padding:6px 8px;font-weight:700;color:var(--accent2);font-size:var(--fs-md)">
            ${r.symbol}
            ${r.eps_accel ? '<span title="EPS Acceleration" style="font-size:var(--fs-2xs);color:var(--green2);margin-left:3px">⚡</span>' : ""}
          </td>
          <td style="padding:6px 8px;font-size:var(--fs-xs);color:var(--text2)">${r.sector || "—"}</td>
          <td style="padding:6px 8px;text-align:right;font-family:var(--mono);font-size:var(--fs-sm)">₹${(r.price||0).toLocaleString("en-IN")}</td>
          <td style="padding:6px 8px;text-align:center">${_scoreCell(r.score, r.tier)}</td>
          <!-- P2-10/11: Final Setup Score (cross-scan consensus) -->
          ${(function() {
            var cs = r.consensus_score || 0;
            var sc = r.scan_count || 1;
            var tl = r.consensus_tier || "—";
            var col = cs >= 60 ? "var(--green)" : cs >= 40 ? "#3b82f6" : cs >= 25 ? "var(--gold)" : "var(--muted)";
            var bg  = cs >= 60 ? "rgba(16,185,129,.15)" : cs >= 40 ? "rgba(59,130,246,.15)" : cs >= 25 ? "rgba(245,158,11,.15)" : "rgba(120,120,120,.10)";
            return `<td style="padding:6px 8px;text-align:center" title="${tl} — appears in ${sc} scans">
                      <span style="display:inline-block;padding:2px 8px;border-radius:10px;background:${bg};color:${col};font-weight:700;font-size:var(--fs-xs)">
                        ${cs.toFixed(0)} · ${sc}×
                      </span>
                    </td>`;
          })()}
          <!-- P2-14: Stage transition badge -->
          ${(function() {
            var sb = r.stage_badge;
            if (!sb) return '<td style="padding:6px 8px;color:var(--muted);font-size:var(--fs-xs)">—</td>';
            var col = r.stage_fresh ? "var(--green)" : "var(--muted)";
            var w = r.stage_fresh ? "700" : "500";
            return `<td style="padding:6px 8px;color:${col};font-weight:${w};font-size:var(--fs-xs)" title="In current stage since ${r.stage_since || '?'}">${sb}${r.stage_fresh ? ' 🟢' : ''}</td>`;
          })()}
          <!-- Factor breakdown mini bars -->
          <td style="padding:4px 6px;text-align:right" title="Quality: ROE · D/E · EPS · Sales growth">${_factorBar(r.q_score,  20, "var(--accent2)")}</td>
          <td style="padding:4px 6px;text-align:right" title="Momentum: 6M/12M rank · RS line">${_factorBar(r.m_score,  20, "var(--green)")}</td>
          <td style="padding:4px 6px;text-align:right" title="Smart Money: Delivery · Bulk deals · F&O OI · FII">${_factorBar(r.sm_score, 25, "var(--gold)")}</td>
          <td style="padding:4px 6px;text-align:right" title="Earnings Quality: EPS accel · Promoter">${_factorBar(r.eq_score, 15, "var(--accent)")}</td>
          <td style="padding:4px 6px;text-align:right" title="Technical: Stage · MA stack · ADX">${_factorBar(r.t_score,  10, "var(--green2)")}</td>
          <td style="padding:4px 6px;text-align:right" title="Risk: Max DD · ADTV · Extension">${_factorBar(r.r_score,  10, "var(--accent2)")}</td>
          <!-- Smart money details -->
          <td style="padding:4px 6px;text-align:right;font-size:var(--fs-xs);color:${_numColour(r.deliv_pct, 55, 30)}">${_fmt(r.deliv_pct,0,"%")}</td>
          <td style="padding:4px 6px;text-align:center;font-size:var(--fs-micro)">${_foBadge(r.fo_signal)}</td>
          <!-- Returns -->
          <td style="padding:4px 6px;text-align:right;font-size:var(--fs-xs);color:${_numColour(r.r1m, 3, -3)}">${_fmt(r.r1m,1,"%")}</td>
          <td style="padding:4px 6px;text-align:right;font-size:var(--fs-xs);color:${_numColour(r.r6m, 10, -5)}">${_fmt(r.r6m,1,"%")}</td>
          <td style="padding:4px 6px;text-align:right;font-size:var(--fs-xs);color:${_numColour(r.rs_6m, 5, -5)}">${_fmt(r.rs_6m,1,"%")}</td>
          <!-- Fundamentals -->
          <td style="padding:4px 6px;text-align:right;font-size:var(--fs-xs);color:${_numColour(r.roe, 15, 8)}">${_fmt(r.roe,1,"%")}</td>
          <td style="padding:4px 6px;text-align:right;font-size:var(--fs-xs);color:${_numColour(r.eps_growth, 15, 0)}">${_fmt(r.eps_growth,0,"%")}</td>
          <td style="padding:4px 6px;text-align:right;font-size:var(--fs-xs)">${_fmt(r.debt_eq,2,"")}</td>
          <!-- Exit signals -->
          <td style="padding:4px 8px">${_exitBadges(r.exit_flags)}</td>
          <!-- Why -->
          <td style="padding:4px 8px;font-size:var(--fs-micro);color:var(--text2);max-width:200px;white-space:normal;line-height:1.4">${r.why || "—"}</td>
        </tr>
        ${exitRow}`;
    }).join("");
  };

  // ── Sort ────────────────────────────────────────────────────────────────────
  window.aeSort = function (col) {
    if (_sortCol === col) {
      _sortDir *= -1;
    } else {
      _sortCol = col;
      _sortDir = col === "tier" ? 1 : -1;
    }
    document.querySelectorAll(".ae-sort").forEach(th => {
      th.classList.remove("sort-asc", "sort-desc");
      if (th.id === "aeTh-" + col) {
        th.classList.add(_sortDir === -1 ? "sort-desc" : "sort-asc");
      }
    });
    aeRender();
  };

  // ── Filter pills ────────────────────────────────────────────────────────────
  window.aeFilter = function (type, val) {
    if (type === "tier")  _tierFilter = val;
    if (type === "exit")  _exitFilter = !_exitFilter;
    if (type === "fo")    _foFilter   = val;

    // Update pill active states
    ["all","buy","strong","avoid"].forEach(v => {
      const el = document.getElementById("aeF-" + v);
      if (el) el.style.background = (_tierFilter === v) ? "rgba(59,130,246,0.25)" : "";
    });
    const exitEl = document.getElementById("aeF-exit");
    if (exitEl) exitEl.style.background = _exitFilter ? "rgba(239,68,68,0.25)" : "";
    ["all","bullish","bearish"].forEach(v => {
      const el = document.getElementById("aeFo-" + v);
      if (el) el.style.background = (_foFilter === v) ? "rgba(245,158,11,0.25)" : "";
    });
    aeRender();
  };

  // ── Search ──────────────────────────────────────────────────────────────────
  window.aeSearch = function () {
    _searchText = (document.getElementById("aeSearch")?.value || "").trim();
    aeRender();
  };

  // ── Scan control ─────────────────────────────────────────────────────────────
  window.aeStartScan = function () {
    document.getElementById("aeEmpty").style.display   = "none";
    document.getElementById("aeResults").style.display = "none";
    document.getElementById("aeProgress").style.display = "block";
    document.getElementById("aeScanBtn").disabled = true;
    document.getElementById("aeScanBtn").textContent = "⏳ Scanning…";
    document.getElementById("aeMeta").textContent = "";

    fetch("/api/alpha/scan", { method: "POST" })
      .then(r => r.json())
      .then(() => {
        if (_polling) clearInterval(_polling);
        _polling = setInterval(_poll, 1800);
      })
      .catch(e => {
        _scanDone(null, "Failed to start: " + e);
      });
  };

  function _poll() {
    fetch("/api/alpha/status")
      .then(r => r.json())
      .then(d => {
        if (d.running) {
          const bar = document.getElementById("aeProgressBar");
          const msg = document.getElementById("aeProgressMsg");
          if (bar) bar.style.width = (d.pct || 0) + "%";
          if (msg) msg.textContent = d.message || "Scanning…";
        } else {
          clearInterval(_polling);
          _polling = null;
          _scanDone(d.result, d.error);
        }
      })
      .catch(() => {});
  }

  function _scanDone(result, err) {
    document.getElementById("aeProgress").style.display = "none";
    document.getElementById("aeScanBtn").disabled = false;
    document.getElementById("aeScanBtn").textContent = "▶ Run Scan";

    if (err || !result) {
      document.getElementById("aeEmpty").style.display = "block";
      document.getElementById("aeEmpty").textContent = "Error: " + (err || "Unknown error");
      return;
    }

    _data = result;

    // Update tier counters
    document.getElementById("aeUniverseCount").textContent = result.total_scanned || "—";
    document.getElementById("aeBuyCount").textContent      = result.buy_count || 0;
    document.getElementById("aeStrongCount").textContent   = result.strong_count || 0;
    document.getElementById("aeMonitorCount").textContent  = result.monitor_count || 0;
    document.getElementById("aeExitCount").textContent     = result.exit_count || 0;
    document.getElementById("aeFOCount").textContent       = result.fo_count || "—";
    document.getElementById("aeFIISignal").textContent     = result.fii_signal || "—";
    document.getElementById("aeFIISignal").style.color     =
      (result.fii_signal||"").includes("Buying")  ? "var(--green)" :
      (result.fii_signal||"").includes("Selling") ? "var(--red)"   : "var(--muted)";

    const ts = result.computed_at ? new Date(result.computed_at * 1000).toLocaleTimeString() : "";
    document.getElementById("aeMeta").textContent = `Computed at ${ts}`;

    document.getElementById("aeResults").style.display = "block";
    aeRender();
  }

  // ── On load: check if cached result exists ───────────────────────────────────
  document.addEventListener("DOMContentLoaded", function () {
    fetch("/api/alpha/status")
      .then(r => r.json())
      .then(d => {
        if (d.result && !d.running) {
          _scanDone(d.result, null);
        } else if (d.running) {
          document.getElementById("aeEmpty").style.display   = "none";
          document.getElementById("aeProgress").style.display = "block";
          document.getElementById("aeScanBtn").disabled = true;
          document.getElementById("aeScanBtn").textContent = "⏳ Scanning…";
          if (_polling) clearInterval(_polling);
          _polling = setInterval(_poll, 1800);
        }
      })
      .catch(() => {});
  });

})();
