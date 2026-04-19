# NSE Trend Screener

A Python + Flask web app to screen NSE stocks for breakout candidates using technical analysis.

## Screening Logic
- **Uptrend filter:** Price > MA50 > MA100 > MA200
- **IBD-style RS Score** (1–99): weighted rank of 3M/6M/12M returns
- Filters: Market Cap, ADTV, % from All-Time High

## Setup

```bash
pip install -r requirements.txt
python3 app.py
```

Open **http://localhost:5050** in your browser.

## Filters
| Filter | Description |
|---|---|
| Min Market Cap (Cr) | Minimum market capitalisation |
| Min ADTV (Cr) | Minimum avg daily trading volume |
| Max % From ATH | How far below all-time high (e.g. -15) |
| Min IBD Score | Relative strength score threshold (60–80 recommended) |

## Setups to look for after screening
VCP · Cup and Handle · ATH Breakout · Rectangular Breakout

## Data Source
Yahoo Finance via `yfinance`. Not financial advice.
