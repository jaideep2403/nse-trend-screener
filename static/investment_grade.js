/* LOCAL-ONLY: Investment-Grade scanner JS. Gitignored. */

let _igData = null;
let _igPoll = null;
let _igFilter = { tier: 'all', trend: 'any', sector: '', search: '' };
let _igSort   = { col: 'score', asc: false };

function igFmt(v, dec=1) {
  if (v == null) return '—';
  return (typeof v === 'number') ? v.toFixed(dec) : String(v);
}
function igFmtPct(v) {
  if (v == null) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
}
function igFmtINR(v) {
  if (v == null) return '—';
  return '₹' + v.toLocaleString('en-IN', {minimumFractionDigits:0, maximumFractionDigits:2});
}
function igPctColor(v) {
  if (v == null) return 'var(--muted)';
  return v >= 0 ? 'var(--green)' : 'var(--red)';
}
function igTierBg(t) {
  if (t.startsWith('A')) return 'rgba(16,185,129,0.18)';
  if (t.startsWith('B')) return 'rgba(245,158,11,0.18)';
  if (t.startsWith('Watchlist')) return 'rgba(100,116,139,0.18)';
  return 'var(--surface2)';
}
function igTierColor(t) {
  if (t.startsWith('A')) return 'var(--green)';
  if (t.startsWith('B')) return 'var(--gold)';
  if (t.startsWith('Watchlist')) return 'var(--text2)';
  return 'var(--text)';
}
function igTierIcon(t) {
  if (t === 'A')           return '🏆';
  if (t === 'B')           return '💎';
  if (t === 'Watchlist')   return '👀';
  if (t === 'A (Tech)')    return '🥈';   // technical-only A
  if (t === 'B (Tech)')    return '⚙️';   // technical-only B
  if (t === 'Watchlist (Tech)') return '🔧';
  return '';
}

function igStartScan() {
  const btn = document.getElementById('igScanBtn');
  btn.disabled = true; btn.textContent = '⏳ Scanning…';
  document.getElementById('igProgress').style.display = 'block';
  document.getElementById('igEmpty').style.display = 'none';
  fetch('/api/investment_grade/scan', {method:'POST'}).then(r => r.json()).then(() => {
    _igPoll = setInterval(igPoll, 1500);
  });
}

function igPoll() {
  fetch('/api/investment_grade/status').then(r => r.json()).then(d => {
    const pb  = document.getElementById('igProgressBar');
    const msg = document.getElementById('igProgressMsg');
    if (d.total > 0) pb.style.width = (d.progress / d.total * 100) + '%';
    if (d.message) msg.textContent = d.message;
    if (!d.running && (d.result || d.error)) {
      clearInterval(_igPoll);
      const btn = document.getElementById('igScanBtn');
      btn.disabled = false; btn.textContent = '🔄 Refresh Scan';
      document.getElementById('igProgress').style.display = 'none';
      if (d.error) {
        document.getElementById('igEmpty').style.display = 'block';
        document.getElementById('igEmpty').textContent = 'Error: ' + d.error;
        return;
      }
      _igData = d.result;
      igRenderInitial();
    }
  });
}

function igRenderInitial() {
  if (!_igData) return;
  const r = _igData;
  document.getElementById('igUniverse').textContent = r.universe_count || '—';
  document.getElementById('igTierA').textContent    = (r.tier_counts && r.tier_counts.A) || 0;
  const ATEl = document.getElementById('igTierAT');
  if (ATEl) ATEl.textContent                         = (r.tier_counts && r.tier_counts.A_tech) || 0;
  document.getElementById('igTierB').textContent    = (r.tier_counts && r.tier_counts.B) || 0;
  document.getElementById('igTierW').textContent    = (r.tier_counts && r.tier_counts.Watchlist) || 0;

  // Sector dropdown
  const sectors = [...new Set((r.stocks || []).map(s => s.sector))].sort();
  const sel = document.getElementById('igSectorFilter');
  sel.innerHTML = '<option value="">All sectors</option>' +
    sectors.map(s => `<option value="${s}">${s}</option>`).join('');

  document.getElementById('igMeta').textContent =
    `${r.qualifying} qualifying · ${r.fund_coverage} with fundamentals · refreshed ${new Date(r.computed_at*1000).toLocaleTimeString()}`;

  document.getElementById('igResults').style.display = 'block';
  igFilter('tier', 'all', false);
  igRender();
}

function igFilter(type, val, doRender=true) {
  if (type === 'tier')  _igFilter.tier  = val;
  if (type === 'trend') _igFilter.trend = (_igFilter.trend === val ? 'any' : val);
  // Update pill highlighting
  ['all','A','AB','watch'].forEach(k => {
    const b = document.getElementById('igF-' + k);
    if (!b) return;
    const on = (_igFilter.tier === k);
    b.style.cssText += `;background:${on?'var(--accent)':'var(--surface2)'};color:${on?'#fff':'var(--text2)'};border-color:${on?'var(--accent)':'var(--border)'}`;
  });
  ['6m','12m'].forEach(k => {
    const b = document.getElementById('igF-' + k);
    if (!b) return;
    const on = (_igFilter.trend === k);
    b.style.cssText += `;background:${on?'var(--gold)':'var(--surface2)'};color:${on?'#000':'var(--text2)'};border-color:${on?'var(--gold)':'var(--border)'}`;
  });
  if (doRender) igRender();
}

function igSort(col) {
  if (_igSort.col === col) {
    _igSort.asc = !_igSort.asc;
  } else {
    _igSort.col = col;
    // Numeric columns default to descending (best first); strings ascending
    const numericCols = ['price','score','months_ma200','r_squared','max_drawdown',
                         'ret_6m','ret_6m_excess','ret_12m','adtv_cr','roe',
                         'eps_growth','sales_growth','debt_eq','pe'];
    _igSort.asc = !numericCols.includes(col);
  }
  igRender();
}

function igRender() {
  if (!_igData) return;
  const sectorFilter = document.getElementById('igSectorFilter').value;
  const search = document.getElementById('igSearch').value.trim().toUpperCase();

  let rows = (_igData.stocks || []).filter(r => {
    if (_igFilter.tier === 'A'     && !r.tier.startsWith('A')) return false;
    if (_igFilter.tier === 'AB'    && !(r.tier.startsWith('A') || r.tier.startsWith('B'))) return false;
    if (_igFilter.tier === 'watch' && !r.tier.startsWith('Watchlist')) return false;
    if (_igFilter.trend === '6m'   && r.months_ma200 < 132) return false;   // ~6 months
    if (_igFilter.trend === '12m'  && r.months_ma200 < 252) return false;   // ~12 months
    if (sectorFilter && r.sector !== sectorFilter) return false;
    if (search && !r.symbol.includes(search)) return false;
    return true;
  });

  // ── Sort ──
  const col = _igSort.col, asc = _igSort.asc;
  const sentinel = asc ? Infinity : -Infinity;
  rows.sort((a, b) => {
    let va = a[col], vb = b[col];
    if (typeof va === 'string' || typeof vb === 'string') {
      va = va == null ? '' : String(va);
      vb = vb == null ? '' : String(vb);
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    va = va == null ? sentinel : va;
    vb = vb == null ? sentinel : vb;
    return asc ? va - vb : vb - va;
  });

  // Update sort indicator on the active header
  document.querySelectorAll('.ig-sort').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
  });
  const activeTh = document.getElementById('igTh-' + col);
  if (activeTh) activeTh.classList.add(asc ? 'sort-asc' : 'sort-desc');

  const tbody = document.getElementById('igTbody');
  document.getElementById('igCount').textContent = rows.length + ' shown';

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="19" style="text-align:center;padding:30px;color:var(--muted)">No stocks match these filters</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map((r, idx) => `
    <tr style="border-bottom:1px solid var(--border)" title="${(r.tech_flags||[]).concat(r.funda_flags||[]).join(' · ')}">
      <td style="padding:7px 10px;color:var(--muted);font-size:var(--fs-xs)">${idx+1}</td>
      <td style="padding:7px 10px"><b style="color:var(--accent)">${r.symbol}</b></td>
      <td style="padding:7px 10px;font-size:var(--fs-xs);color:var(--text2)">${r.sector || '—'}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono)">${igFmtINR(r.price)}</td>
      <td style="padding:7px 10px;text-align:center;font-weight:800;color:${igTierColor(r.tier)}">${r.score}/${r.max_score}</td>
      <td style="padding:7px 10px;text-align:center">
        <span style="display:inline-block;padding:2px 8px;background:${igTierBg(r.tier)};color:${igTierColor(r.tier)};border-radius:10px;font-size:var(--fs-micro);font-weight:800">${igTierIcon(r.tier)} ${r.tier}</span>
      </td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${r.months_label >= 6 ? 'var(--green)' : 'var(--text2)'}">${r.months_label}m</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${(r.r_squared||0) >= 0.5 ? 'var(--green)' : 'var(--text2)'}">${igFmt(r.r_squared, 2)}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${r.max_drawdown < 15 ? 'var(--green)' : r.max_drawdown < 25 ? 'var(--gold)' : 'var(--red)'}">−${igFmt(r.max_drawdown, 0)}%</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${igPctColor(r.ret_6m)}">${igFmtPct(r.ret_6m)}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${igPctColor(r.ret_6m_excess)}">${igFmtPct(r.ret_6m_excess)}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${igPctColor(r.ret_12m)}">${igFmtPct(r.ret_12m)}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${(r.adtv_cr||0) >= 50 ? 'var(--green)' : 'var(--text2)'}">${igFmt(r.adtv_cr, 0)} Cr</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${(r.roe||0) >= 15 ? 'var(--green)' : 'var(--text2)'}">${r.roe != null ? r.roe.toFixed(0)+'%' : '—'}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${igPctColor(r.eps_growth)}">${r.eps_growth != null ? igFmtPct(r.eps_growth) : '—'}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${igPctColor(r.sales_growth)}">${r.sales_growth != null ? igFmtPct(r.sales_growth) : '—'}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:${(r.debt_eq||99) < 1 ? 'var(--green)' : 'var(--red)'}">${r.debt_eq != null ? r.debt_eq.toFixed(2) : '—'}</td>
      <td style="padding:7px 10px;text-align:right;font-family:var(--mono);color:var(--text2)">${r.pe != null ? r.pe.toFixed(0) : '—'}</td>
      <td style="padding:7px 10px;font-size:var(--fs-micro);color:var(--text2);max-width:340px">${(r.tech_flags||[]).concat(r.funda_flags||[]).slice(0,5).join(' · ')}</td>
    </tr>
  `).join('');
}

// Auto-load when tab opened
(function hookIGTab() {
  const orig = window.showTab;
  if (typeof orig !== 'function') return;
  window.showTab = function(name, btn) {
    orig(name, btn);
    if (name === 'investgrade' && _igData) igRender();
    // Auto-fetch existing cache silently on first open
    if (name === 'investgrade' && !_igData) {
      fetch('/api/investment_grade/status').then(r => r.json()).then(d => {
        if (d.result && !d.running) {
          _igData = d.result;
          igRenderInitial();
        }
      });
    }
  };
})();
