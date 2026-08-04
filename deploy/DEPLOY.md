# Deploying Ascent Wealth Labs to EC2

Decisions this reflects: **owner-only tabs stay off the box**, **EC2 owns NSE
fetching**, **Route53 DNS only — no Cloudflare**, so TLS terminates in nginx.

---

## 0. Instance

| | |
|---|---|
| Type | **t3.medium (4 GB)** — `gunicorn.conf.py` notes 150–300 MB resident per worker, and a single process was measured at **312 MB RSS** during a cache rebuild. t3.small (2 GB) leaves no headroom for the rebuild. |
| Disk | 20 GB gp3 — the bhavcopy cache is **214 MB** today and grows ~170 KB/trading day. |
| Ports | Security group: **80 + 443 only**. Never expose 5050 — gunicorn binds to loopback. |
| OS | Amazon Linux 2023 / Ubuntu 24.04, Python 3.13 |

---

## 1. Ship the cache BEFORE first boot ← most important step

A clean box will try to pull **1,583 bhavcopy day files (~6 years, 214 MB)** from
NSE in one burst, from a fresh datacenter IP. That is the exact "get us blocked"
scenario. Copy it instead:

```bash
ssh ec2 'sudo mkdir -p /var/lib/ascent && sudo chown ascent:ascent /var/lib/ascent'
rsync -avz --progress /tmp/nse_bhav_days/  ec2:/var/lib/ascent/nse_bhav_days/
rsync -avz --progress /tmp/nse_ohlcv_pkl/  ec2:/var/lib/ascent/nse_ohlcv_pkl/
```

Note the paths move off `/tmp` — it is cleared on reboot and tmpfs on some AMIs.
`data_fetcher.py` already reads `BHAV_DIR` / `OHLCV_DIR` from the environment, so
this needs no code change; the systemd unit sets them.

## 2. Code

```bash
rsync -av --exclude-from=deploy/rsync-exclude.txt ./ ec2:/opt/ascent/
```

**Use rsync, not `git clone`.** Four templates the public app needs are gitignored
(`alpha_engine_tab.html`, `investment_grade_tab.html` and their JS). Because the
includes are `ignore missing`, a git-only deploy drops those tabs **silently** —
no error, they simply don't render.

`.auth_users.json` (your hashed logins) is **not** excluded, so it travels with the
rsync above — just tighten its mode afterwards, since rsync preserves the local one
but it is worth asserting:

```bash
ssh ec2 'chmod 600 /opt/ascent/.auth_users.json'
```

`.auth_secret` **is** excluded on purpose — the box generates its own. Sharing a
session-signing key across hosts has no upside.

Verified by dry-run (`rsync -avn --exclude-from=deploy/rsync-exclude.txt`): 151
files ship; `portfolio_tab.html`, `strategy_tab.html`, `portfolio.js`,
`.portfolio.json`, `.strategy_journal.db` and `.auth_secret` are all blocked, while
`alpha_engine_tab.html`, `investment_grade_tab.html`, `alpha_engine.js` and
`.auth_users.json` all pass through.

## 3. Python

```bash
ssh ec2
sudo useradd -r -s /usr/sbin/nologin -d /opt/ascent ascent
sudo chown -R ascent:ascent /opt/ascent /var/lib/ascent
cd /opt/ascent && python3 -m venv venv
./venv/bin/pip install -r requirements.txt      # 6 packages, nothing new added
```

The hardening in `security.py` is stdlib + Werkzeug only — no new dependency to
install or audit.

## 4. Service

```bash
sudo cp deploy/ascent.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ascent
journalctl -u ascent -f          # expect: [security] hardening active (production=True)
```

## 5. nginx + TLS

```bash
sudo cp deploy/nginx-ascent.conf /etc/nginx/conf.d/ascent.conf
sudo sed -i 's/YOUR_DOMAIN/<your Route53 name>/g' /etc/nginx/conf.d/ascent.conf
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d <your Route53 name>
```

Point the Route53 A record at the Elastic IP **before** running certbot — the
HTTP-01 challenge resolves the name from the public internet.

---

## 6. Verify hardening actually landed

```bash
curl -sI https://YOUR_DOMAIN/login | grep -iE 'strict-transport|x-frame|x-content|content-security'
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://YOUR_DOMAIN/api/bhavcopy/refresh   # expect 403
for i in $(seq 1 12); do curl -s -o /dev/null -w '%{http_code} ' \
  -d 'username=x&password=y' https://YOUR_DOMAIN/login; done; echo                          # expect 429s
curl -sI https://YOUR_DOMAIN/login | grep -i 'set-cookie'                                   # expect Secure; HttpOnly; SameSite=Lax
```

Measured locally with `ASCENT_ENV=production`:

| check | result |
|---|---|
| `SESSION_COOKIE_SECURE` | True |
| ProxyFix installed | True |
| POST, no Origin | 403 |
| POST, foreign Origin | 403 |
| POST, matching Origin | 200 |
| GET, no Origin | 200 (safe method, unaffected) |
| HSTS over https | `max-age=31536000; includeSubDomains` |
| 11 bad logins | 8×200 then 429 |

---

## Still open — decide before you take subscribers

These are **not** done and I have not touched them:

1. **Backups.** `.auth_users.json` and the cache have no backup. An instance
   replacement loses your logins and re-triggers the 1,583-file NSE download.
2. **The demo password is public** (`demo/demo123`, printed on the sign-in page).
   Fine while it's a demo; rotate it if the box gets real traffic.
3. **No monitoring.** Nothing alerts if the bhavcopy scheduler stops fetching. The
   `[STUCK — escalating]` log line is the only signal and nobody reads it.
4. **One box, no redundancy.** Acceptable for EOD analytics; worth stating.
