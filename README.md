# Trading Reporter v4.0

A Python-based equity research tool built with pandas, NumPy, and yfinance. Includes full technical indicator support (RSI, MACD, Bollinger Bands, moving averages), multi-source news aggregation, and valuation models (DCF, EPV, and multiples).

Features an automated 1–10 scoring framework that generates BUY/HOLD/SELL recommendations.

Integrated Google Gemini API for AI-driven price-event interpretation with robust retry/backoff logic.

Also supports trading signal detection (Golden Cross, Death Cross, 52-week breakouts) and sector benchmarking against major ETF indices.

---

## Features

| Feature | Description |
|---------|-------------|
| **Technical Indicators** | RSI, MACD, Bollinger Bands, 50/200-day MA, ATR, Support/Resistance |
| **Multi-Source News** | Aggregates from Yahoo Finance, Google RSS, and Finviz |
| **Event Detection** | Detects gaps, volume spikes, earnings moves; matches news to price events |
| **Valuation Models** | DCF, EPV, Multiples with weighted composite fair value |
| **Scoring System** | 1-10 score combining valuation, technicals, momentum, fundamentals, sector |
| **Trading Signals** | Golden Cross, Death Cross, RSI Oversold/Overbought, MACD Crossovers, 52W Breakouts |
| **Sector Comparison** | Benchmarks against sector ETFs (XLK, XLF, XLV, etc.) |
| **AI Analysis** | Optional Gemini API for price event interpretation |

---

## Installation

### 1. Install Python Dependencies

```bash
pip install yfinance pandas numpy requests google-genai
```

### 2. Get Gemini API Key (Optional - for AI features)

1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key (starts with `AIzaSy...`)

---

## Usage

### Basic Usage (Rule-Based Analysis)

```bash
python trading_reporter_v4.py AAPL
```

### With AI Analysis (Requires API Key)

**PowerShell:**
```powershell
$env:GEMINI_API_KEY="AIzaSy_your_key_here"
python trading_reporter_v4.py AAPL
```

**Command Prompt:**
```cmd
set GEMINI_API_KEY=AIzaSy_your_key_here
python trading_reporter_v4.py AAPL
```

**Mac/Linux:**
```bash
export GEMINI_API_KEY=AIzaSy_your_key_here
python trading_reporter_v4.py AAPL
```

### Multiple Stocks

```bash
python trading_reporter_v4.py AAPL NVDA MSFT GOOG
```

### Skip AI (Force Rule-Based Only)

```bash
python trading_reporter_v4.py AAPL --no-ai
```

### Quiet Mode (Minimal Output)

```bash
python trading_reporter_v4.py AAPL -q
```

### Custom Throttle Between Tickers

```bash
python trading_reporter_v4.py AAPL NVDA MSFT --throttle 3.0
```

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `TICKER` | Stock ticker symbol(s) to analyze |
| `--no-ai` | Skip AI analysis, use rule-based only |
| `-q, --quiet` | Minimal output |
| `--debug` | Enable debug mode |
| `--throttle N` | Delay N seconds between tickers (default: 1.5) |

---

## Output

Reports are saved to the `reports/` folder with the format:
```
reports/AAPL_REPORT_20260201_163000.txt
```

### Sample Output

```
============================================================
Analyzing AAPL...
============================================================
[1/9] Fetching price data...
  ✓ 400 days
[2/9] Fetching fundamentals...
  ✓ Apple Inc.
[3/9] Fetching news (Yahoo + Google + Finviz)...
  ✓ 15 articles
[4/9] Fetching earnings...
  ✓ 4 earnings records
[5/9] Detecting price events...
  ✓ 5 events detected
[6/9] Calculating technicals (RSI, MACD, BB, MA)...
  ✓ 3 signals
[7/9] Comparing to sector...
  ✓ UNDERPERFORMING vs XLK
[8/9] Running valuation...
  ✓ $180.67 (SIGNIFICANTLY OVERVALUED)
[9/9] Generating analysis...
  ✓ HOLD (Score: 5.5/10)

================================================================================
AAPL: $259.48 → Fair $180.67 | Score: 5.5/10 | HOLD
Signals: RSI_OVERBOUGHT, MACD_BEARISH
================================================================================

✓ Report saved: reports/AAPL_REPORT_20260201_163000.txt
```

---

## Report Contents

Each generated report includes:

- **Score & Recommendation** - Overall 1-10 score with BUY/HOLD/SELL
- **Component Scores** - Valuation, Technicals, Momentum, Fundamentals, Sector
- **AI Analysis** - Price event interpretation (if API enabled)
- **Trading Signals** - Golden Cross, Death Cross, RSI alerts, etc.
- **Technical Indicators** - RSI, MACD, Bollinger Bands, Moving Averages
- **Performance** - 1D, 7D, 30D, 90D returns with volatility
- **Sector Comparison** - Stock vs sector ETF performance
- **Price Events** - Detected moves with matched news
- **Recent News** - Headlines from 3 sources
- **Valuation** - DCF, EPV, Multiples with fair value range
- **Fundamentals** - P/E, P/B, ROE, margins, growth, debt

---

## Troubleshooting

### "google-genai not installed"
```bash
pip install google-genai
```

### "Gemini 429 / Rate Limited"
- Wait 1-2 minutes and try again
- Use a different Google account for new API key
- Use `--no-ai` flag to skip AI

### "No price data"
- Check if ticker symbol is correct
- Some tickers may not be available on Yahoo Finance

---

## Project Structure

```
Trading Reporter v4.0/
├── trading_reporter_v4.py    # Main script
├── README.md                 # This file
├── reports/                  # Generated reports (auto-created)
└── cache/                    # API cache (auto-created)
```

---

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **NumPy** - Numerical calculations
- **yfinance** - Stock data from Yahoo Finance
- **requests** - HTTP requests for news
- **google-genai** - Gemini API integration

---

## DISCLAIMER

**Educational purposes only. NOT financial advice.**

This tool is for learning and research purposes. Do not make investment decisions based solely on this tool's output. Always do your own research and consult with a qualified financial advisor.

---

## License

MIT License
