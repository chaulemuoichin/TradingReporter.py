#!/usr/bin/env python3
"""
================================================================================
TRADING REPORTER v3.0 - PRODUCTION HARDENED
================================================================================
Changes from v2.1:
    ✓ Single Gemini API call per stock (MAX 1)
    ✓ Exponential backoff + jitter for 429/RESOURCE_EXHAUSTED
    ✓ Response caching (cache/gemini_{ticker}_{date}.json)
    ✓ Graceful fallback to RULE_BASED mode
    ✓ Throttling between multiple tickers
    ✓ Improved prompt for consistent output
    ✓ Debug mode for diagnostics
    ✓ Never crashes - always produces report

Install:
    pip install yfinance pandas numpy requests google-genai

Usage:
    python trading_reporter.py NVDA
    python trading_reporter.py AAPL MSFT GOOG --throttle 2.0
    python trading_reporter.py NVDA --no-ai
    python trading_reporter.py NVDA --debug
    python trading_reporter.py NVDA --no-cache

Environment Variables:
    GEMINI_API_KEY - Your API key from https://aistudio.google.com/app/apikey
================================================================================
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not installed - run: pip install yfinance")

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed - run: pip install google-genai")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """Retry/backoff configuration for Gemini API"""
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 20.0
    jitter_min: float = 0.7
    jitter_max: float = 1.3

@dataclass
class ValuationConfig:
    """Valuation parameters"""
    risk_free_rate: float = 0.045
    equity_risk_premium: float = 0.055
    default_tax_rate: float = 0.21
    cost_of_debt_spread: float = 0.02
    dcf_growth_short: float = 0.08
    dcf_growth_terminal: float = 0.025
    dcf_forecast_years: int = 5
    wacc_floor: float = 0.05
    wacc_cap: float = 0.20
    fair_pe_multiple: float = 20.0
    fair_pb_multiple: float = 3.5
    fair_ev_ebitda_multiple: float = 12.0
    epv_required_return: float = 0.10
    dcf_weight: float = 0.40
    multiples_weight: float = 0.35
    epv_weight: float = 0.25

@dataclass
class Config:
    """Main configuration"""
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv('GEMINI_API_KEY'))
    enable_gemini: bool = field(default_factory=lambda: os.getenv('ENABLE_GEMINI', 'true').lower() == 'true')
    output_dir: str = 'reports'
    cache_dir: str = 'cache'
    performance_windows: List[int] = field(default_factory=lambda: [1, 7, 30])
    price_history_days: int = 365
    news_lookback_days: int = 30
    valuation: ValuationConfig = field(default_factory=ValuationConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    z_threshold: float = 3.0
    min_abs_return: float = 0.03
    min_volume_ratio: float = 2.0
    rolling_window: int = 20
    news_window_minutes: List[int] = field(default_factory=lambda: [60, 360, 1440])
    max_news_per_event: int = 3
    default_throttle: float = 1.5
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def ai_enabled(self) -> bool:
        return bool(self.enable_gemini and self.gemini_api_key and GEMINI_AVAILABLE)

CONFIG = Config()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if pd.isna(value) or np.isinf(value):
            return default
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(',', '').replace('$', ''))
        except ValueError:
            return default
    return default

def ensure_shares_valid(info: Dict) -> Optional[float]:
    """Validate shares_outstanding"""
    shares = safe_number(info.get('shares_outstanding'))
    price = safe_number(info.get('current_price'))
    market_cap = safe_number(info.get('market_cap'))
    if not shares:
        if price and market_cap and price > 0:
            return market_cap / price
        return None
    return shares

def compute_hash(text: str) -> str:
    """Compute MD5 hash for caching"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]

# =============================================================================
# GEMINI CACHE
# =============================================================================

class GeminiCache:
    """
    File-based cache for Gemini responses.
    Path: cache/gemini_{ticker}_{date}.json
    """
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _path(self, ticker: str) -> Path:
        date_str = datetime.now().strftime('%Y%m%d')
        return self.cache_dir / f"gemini_{ticker}_{date_str}.json"
    
    def get(self, ticker: str, prompt_hash: str) -> Optional[Dict]:
        """Get cached response if exists and hash matches"""
        path = self._path(ticker)
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('prompt_hash') == prompt_hash:
                return data.get('response')
            return None
        except:
            return None
    
    def set(self, ticker: str, prompt_hash: str, response: Dict, model: str):
        """Save response to cache"""
        path = self._path(ticker)
        data = {
            'ticker': ticker,
            'prompt_hash': prompt_hash,
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'response': response
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def clear(self, ticker: str = None):
        """Clear cache"""
        pattern = f"gemini_{ticker}_*.json" if ticker else "gemini_*.json"
        for f in self.cache_dir.glob(pattern):
            f.unlink()

# =============================================================================
# GEMINI CLIENT WITH RETRY/BACKOFF
# =============================================================================

class GeminiClient:
    """
    Gemini API client with:
    - Exponential backoff + jitter for 429
    - Single call design
    - Caching
    - Graceful fallback
    """
    
    RETRYABLE_ERRORS = ['429', 'RESOURCE_EXHAUSTED', 'Too Many Requests', 'rate limit', 'quota']
    
    def __init__(self, api_key: str, retry_cfg: RetryConfig = None, use_cache: bool = True):
        self.api_key = api_key
        self.retry_cfg = retry_cfg or RetryConfig()
        self.use_cache = use_cache
        self.cache = GeminiCache(CONFIG.cache_dir)
        self.client = None
        self.available = False
        self.error_msg = ""
        self.stats = {'api_calls': 0, 'cache_hits': 0, 'retries': 0, 'fallbacks': 0}
        
        self._init()
    
    def _init(self):
        """Initialize client"""
        if not GEMINI_AVAILABLE:
            self.error_msg = "google-genai not installed"
            return
        if not self.api_key:
            self.error_msg = "GEMINI_API_KEY not found. Running in RULE_BASED mode."
            return
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.available = True
        except Exception as e:
            self.error_msg = f"Gemini init failed: {str(e)[:50]}"
    
    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable"""
        err = str(error).lower()
        return any(e.lower() in err for e in self.RETRYABLE_ERRORS)
    
    def _delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + jitter"""
        cfg = self.retry_cfg
        delay = min(cfg.max_delay, cfg.base_delay * (2 ** attempt))
        return delay * random.uniform(cfg.jitter_min, cfg.jitter_max)
    
    def generate(self, prompt: str, ticker: str, model: str = "gemini-2.0-flash") -> Optional[str]:
        """
        Generate with retry/backoff.
        Returns response text or None.
        """
        if not self.available:
            return None
        
        # Check cache
        prompt_hash = compute_hash(prompt)
        if self.use_cache:
            cached = self.cache.get(ticker, prompt_hash)
            if cached:
                self.stats['cache_hits'] += 1
                logger.info(f"  ✓ Cache HIT for {ticker}")
                return cached.get('text')
        
        # Try with retries
        cfg = self.retry_cfg
        for attempt in range(cfg.max_retries):
            try:
                self.stats['api_calls'] += 1
                response = self.client.models.generate_content(model=model, contents=prompt)
                
                if response and response.text:
                    if self.use_cache:
                        self.cache.set(ticker, prompt_hash, {'text': response.text}, model)
                    return response.text
                return None
                
            except Exception as e:
                if self._is_retryable(e) and attempt < cfg.max_retries - 1:
                    self.stats['retries'] += 1
                    delay = self._delay(attempt)
                    logger.warning(f"  Gemini 429 RESOURCE_EXHAUSTED. Retry {attempt+1}/{cfg.max_retries} in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"  Gemini error: {str(e)[:80]}")
                    break
        
        self.stats['fallbacks'] += 1
        return None

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_price_data(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data"""
    if not YF_AVAILABLE:
        return None
    try:
        stock = yf.Ticker(ticker)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 15)
        df = stock.history(start=start, end=end, interval='1d')
        if df.empty:
            return None
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        df = df.sort_index()
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        return df
    except Exception as e:
        logger.error(f"Price fetch error: {e}")
        return None

def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """Fetch fundamentals"""
    if not YF_AVAILABLE:
        return {}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = {
            'ticker': ticker,
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': safe_number(info.get('currentPrice')) or safe_number(info.get('regularMarketPrice')) or safe_number(info.get('previousClose')),
            'fifty_two_week_high': safe_number(info.get('fiftyTwoWeekHigh')),
            'fifty_two_week_low': safe_number(info.get('fiftyTwoWeekLow')),
            'beta': safe_number(info.get('beta'), default=1.0),
            'market_cap': safe_number(info.get('marketCap')),
            'shares_outstanding': safe_number(info.get('sharesOutstanding')),
            'pe_trailing': safe_number(info.get('trailingPE')),
            'pe_forward': safe_number(info.get('forwardPE')),
            'pb_ratio': safe_number(info.get('priceToBook')),
            'profit_margin': safe_number(info.get('profitMargins')),
            'roe': safe_number(info.get('returnOnEquity')),
            'roa': safe_number(info.get('returnOnAssets')),
            'revenue_growth': safe_number(info.get('revenueGrowth')),
            'earnings_growth': safe_number(info.get('earningsGrowth')),
            'debt_to_equity': safe_number(info.get('debtToEquity')),
            'total_debt': safe_number(info.get('totalDebt')),
            'total_cash': safe_number(info.get('totalCash')),
            'free_cash_flow': safe_number(info.get('freeCashflow')),
            'eps_trailing': safe_number(info.get('trailingEps')),
            'book_value': safe_number(info.get('bookValue')),
            'ebitda': safe_number(info.get('ebitda')),
            'interest_expense': safe_number(info.get('interestExpense')),
            'target_mean': safe_number(info.get('targetMeanPrice')),
            'recommendation': info.get('recommendationKey'),
        }
        data['shares_outstanding'] = ensure_shares_valid(data)
        return data
    except Exception as e:
        logger.error(f"Fundamentals error: {e}")
        return {}

def fetch_earnings_history(ticker: str) -> List[Dict]:
    """Fetch earnings"""
    if not YF_AVAILABLE:
        return []
    try:
        stock = yf.Ticker(ticker)
        earnings = []
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            for idx, row in ed.head(8).iterrows():
                dt = pd.to_datetime(idx)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                earnings.append({
                    'date_str': dt.strftime('%Y-%m-%d'),
                    'eps_estimate': safe_number(row.get('EPS Estimate')),
                    'eps_actual': safe_number(row.get('Reported EPS')),
                    'surprise_pct': safe_number(row.get('Surprise(%)')),
                })
        return earnings
    except:
        return []

def fetch_upcoming_events(ticker: str) -> Dict[str, str]:
    """Fetch upcoming events"""
    if not YF_AVAILABLE:
        return {}
    events = {}
    now = datetime.now(timezone.utc)
    try:
        stock = yf.Ticker(ticker)
        ed = stock.earnings_dates
        if ed is not None and not ed.empty:
            for idx in ed.index:
                dt = pd.to_datetime(idx)
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                if dt > now:
                    events['Next Earnings'] = dt.strftime('%Y-%m-%d')
                    break
        return events
    except:
        return {}

def fetch_news(ticker: str, days: int = 30) -> List[Dict]:
    """Fetch news"""
    if not YF_AVAILABLE:
        return []
    news_list = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
        for item in news:
            pub = item.get('providerPublishTime')
            if not pub:
                continue
            dt = pd.Timestamp(pub, unit='s', tz='UTC')
            if dt < cutoff:
                continue
            news_list.append({
                'title': item.get('title', ''),
                'publisher': item.get('publisher', 'Unknown'),
                'timestamp': pub,
                'dt': dt,
                'datetime_str': dt.strftime('%Y-%m-%d %H:%M UTC'),
            })
    except:
        pass
    return sorted(news_list, key=lambda x: x['timestamp'], reverse=True)[:10]

def match_news_to_events(events: List[Dict], news: List[Dict]) -> List[Dict]:
    """Match news to events"""
    if not events or not news:
        return events
    for event in events:
        event_ts = event.get('timestamp')
        if event_ts is None:
            continue
        if not isinstance(event_ts, pd.Timestamp):
            event_ts = pd.Timestamp(event_ts, tz='UTC')
        matched = []
        for window in CONFIG.news_window_minutes:
            for n in news:
                news_dt = n.get('dt')
                if news_dt is None:
                    continue
                diff = abs((news_dt - event_ts).total_seconds() / 60)
                if diff <= window:
                    matched.append({
                        'title': n['title'],
                        'publisher': n['publisher'],
                        'datetime_str': n.get('datetime_str', ''),
                        'time_delta_min': round(diff, 1),
                    })
            if matched:
                break
        event['matched_news'] = matched[:CONFIG.max_news_per_event]
    return events

# =============================================================================
# ANALYTICS
# =============================================================================

def analyze_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance"""
    if df is None or df.empty:
        return {}
    df = df.sort_index()
    current = float(df['close'].iloc[-1])
    results = {'current_price': round(current, 2), 'windows': {}}
    for w in CONFIG.performance_windows:
        if len(df) < w + 1:
            continue
        start = float(df['close'].iloc[-1 - w])
        ret = (current / start - 1) if start > 0 else 0
        wdf = df.iloc[-w:]
        log_ret = wdf['log_return'].dropna()
        vol = float(log_ret.std()) * np.sqrt(252) if len(log_ret) >= 2 else None
        results['windows'][w] = {
            'return_pct': round(ret * 100, 2),
            'vol_ann_pct': round(vol * 100, 2) if vol else None,
            'high': round(float(wdf['high'].max()), 2),
            'low': round(float(wdf['low'].min()), 2),
        }
    return results

def detect_events(df: pd.DataFrame) -> List[Dict]:
    """Detect price events"""
    if df is None or len(df) < 25:
        return []
    df = df.copy().sort_index()
    df['ret_std'] = df['return'].rolling(CONFIG.rolling_window).std().replace(0, np.nan)
    df['z'] = df['return'] / df['ret_std']
    events = []
    for i in range(CONFIG.rolling_window, len(df)):
        row = df.iloc[i]
        ret = row.get('return', 0)
        z = row.get('z', 0)
        vol_ratio = row.get('volume_ratio', 1)
        if pd.isna(ret) or pd.isna(z):
            continue
        if abs(z) >= CONFIG.z_threshold or (abs(ret) >= CONFIG.min_abs_return and vol_ratio >= CONFIG.min_volume_ratio):
            ts = row.name
            if not isinstance(ts, pd.Timestamp):
                ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            events.append({
                'timestamp': ts,
                'date_str': ts.strftime('%Y-%m-%d'),
                'type': 'surge' if ret > 0 else 'drop',
                'return_pct': round(ret * 100, 2),
                'matched_news': [],
            })
    return sorted(events, key=lambda x: abs(x['return_pct']), reverse=True)[:5]

# =============================================================================
# VALUATION
# =============================================================================

def calculate_wacc(info: Dict) -> Tuple[float, Dict]:
    """Calculate WACC"""
    cfg = CONFIG.valuation
    beta = max(0.3, min(safe_number(info.get('beta'), 1.0), 3.0))
    coe = cfg.risk_free_rate + beta * cfg.equity_risk_premium
    mcap = safe_number(info.get('market_cap'))
    debt = safe_number(info.get('total_debt'), 0)
    if mcap and mcap > 0:
        e_wt = mcap / (mcap + debt)
        d_wt = debt / (mcap + debt)
    else:
        e_wt, d_wt = 0.8, 0.2
    d_wt = min(d_wt, 0.5)
    e_wt = 1 - d_wt
    interest = safe_number(info.get('interest_expense'))
    if interest and debt and debt > 0:
        cod_pre = abs(interest) / debt
    else:
        cod_pre = cfg.risk_free_rate + cfg.cost_of_debt_spread
    cod_post = cod_pre * (1 - cfg.default_tax_rate)
    wacc = e_wt * coe + d_wt * cod_post
    wacc = max(cfg.wacc_floor, min(cfg.wacc_cap, wacc))
    return wacc, {'wacc': round(wacc, 4), 'beta': beta}

def dcf_valuation(info: Dict) -> Dict:
    """DCF valuation"""
    cfg = CONFIG.valuation
    fcf = safe_number(info.get('free_cash_flow'))
    shares = safe_number(info.get('shares_outstanding'))
    if not fcf or fcf <= 0 or not shares or shares <= 0:
        return {'intrinsic_value': None}
    debt = safe_number(info.get('total_debt'), 0)
    cash = safe_number(info.get('total_cash'), 0)
    net_debt = debt - cash
    wacc, _ = calculate_wacc(info)
    g_s, g_t = cfg.dcf_growth_short, cfg.dcf_growth_terminal
    if wacc <= g_t:
        wacc = g_t + 0.03
    pv = 0
    cf = fcf
    for y in range(1, cfg.dcf_forecast_years + 1):
        cf *= (1 + g_s)
        pv += cf / ((1 + wacc) ** y)
    tv = cf * (1 + g_t) / (wacc - g_t)
    pv_tv = tv / ((1 + wacc) ** cfg.dcf_forecast_years)
    ev = pv + pv_tv
    eq = ev - net_debt
    iv = eq / shares
    cp = safe_number(info.get('current_price'), 0)
    up = ((iv / cp) - 1) * 100 if cp > 0 else None
    return {
        'method': 'DCF',
        'intrinsic_value': round(iv, 2),
        'upside_pct': round(up, 1) if up else None,
        'assumptions': {'wacc': round(wacc * 100, 2), 'g_short': round(g_s * 100, 2), 'g_terminal': round(g_t * 100, 2)},
    }

def multiples_valuation(info: Dict) -> Dict:
    """Multiples valuation"""
    cfg = CONFIG.valuation
    cp = safe_number(info.get('current_price'), 0)
    shares = safe_number(info.get('shares_outstanding'))
    debt = safe_number(info.get('total_debt'), 0)
    cash = safe_number(info.get('total_cash'), 0)
    net_debt = debt - cash
    implied = []
    eps = safe_number(info.get('eps_trailing'))
    if eps and eps > 0:
        implied.append({'metric': 'P/E', 'price': round(eps * cfg.fair_pe_multiple, 2)})
    bv = safe_number(info.get('book_value'))
    if bv and bv > 0:
        implied.append({'metric': 'P/B', 'price': round(bv * cfg.fair_pb_multiple, 2)})
    ebitda = safe_number(info.get('ebitda'))
    if ebitda and ebitda > 0 and shares and shares > 0:
        ev = ebitda * cfg.fair_ev_ebitda_multiple
        p = (ev - net_debt) / shares
        if p > 0:
            implied.append({'metric': 'EV/EBITDA', 'price': round(p, 2)})
    target = safe_number(info.get('target_mean'))
    if target:
        implied.append({'metric': 'Analyst', 'price': round(target, 2)})
    if not implied:
        return {'intrinsic_value': None}
    prices = [v['price'] for v in implied if v['price'] > 0]
    med = float(np.median(prices)) if prices else None
    return {
        'method': 'Multiples',
        'intrinsic_value': round(med, 2) if med else None,
        'range_low': round(min(prices), 2) if prices else None,
        'range_high': round(max(prices), 2) if prices else None,
        'implied': implied,
    }

def epv_valuation(info: Dict) -> Dict:
    """EPV valuation"""
    eps = safe_number(info.get('eps_trailing'))
    if not eps or eps <= 0:
        return {'intrinsic_value': None}
    epv = eps / CONFIG.valuation.epv_required_return
    cp = safe_number(info.get('current_price'), 0)
    return {
        'method': 'EPV',
        'intrinsic_value': round(epv, 2),
        'upside_pct': round((epv / cp - 1) * 100, 1) if cp > 0 else None,
    }

def comprehensive_valuation(info: Dict) -> Dict:
    """Run all valuations"""
    cfg = CONFIG.valuation
    cp = safe_number(info.get('current_price'), 0)
    dcf = dcf_valuation(info)
    mult = multiples_valuation(info)
    epv = epv_valuation(info)
    values = []
    methods = {'dcf': dcf, 'multiples': mult, 'epv': epv}
    if dcf.get('intrinsic_value') and dcf['intrinsic_value'] > 0:
        values.append(('DCF', dcf['intrinsic_value'], cfg.dcf_weight))
    if mult.get('intrinsic_value') and mult['intrinsic_value'] > 0:
        values.append(('Multiples', mult['intrinsic_value'], cfg.multiples_weight))
    if epv.get('intrinsic_value') and epv['intrinsic_value'] > 0:
        values.append(('EPV', epv['intrinsic_value'], cfg.epv_weight))
    if not values:
        return {'current_price': cp, 'intrinsic_value': None, 'verdict': 'INSUFFICIENT DATA', 'confidence': 'LOW', 'methods': methods}
    tot_wt = sum(w for _, _, w in values)
    weighted = sum(v * w for _, v, w in values) / tot_wt
    all_v = [v for _, v, _ in values]
    r_lo, r_hi = min(all_v), max(all_v)
    up = ((weighted / cp) - 1) * 100 if cp > 0 else 0
    if up > 25: verdict = 'SIGNIFICANTLY UNDERVALUED'
    elif up > 10: verdict = 'UNDERVALUED'
    elif up > -10: verdict = 'FAIRLY VALUED'
    elif up > -25: verdict = 'OVERVALUED'
    else: verdict = 'SIGNIFICANTLY OVERVALUED'
    spread = (r_hi - r_lo) / weighted if weighted > 0 else 1
    conf = 'HIGH' if spread < 0.30 else ('MEDIUM' if spread < 0.50 else 'LOW')
    return {
        'current_price': round(cp, 2),
        'intrinsic_value': round(weighted, 2),
        'range_low': round(r_lo, 2),
        'range_high': round(r_hi, 2),
        'upside_pct': round(up, 1),
        'verdict': verdict,
        'confidence': conf,
        'components': [{'method': m, 'value': round(v, 2), 'weight': w} for m, v, w in values],
        'methods': methods,
    }

# =============================================================================
# AI ANALYZER - SINGLE CALL DESIGN
# =============================================================================

class AIAnalyzer:
    """
    AI Analyzer with:
    - SINGLE Gemini call per stock (MAX 1)
    - Caching
    - Retry/backoff
    - Graceful fallback
    """
    
    def __init__(self, use_cache: bool = True, debug: bool = False):
        self.debug = debug
        self.use_cache = use_cache
        self.client = None
        self.available = False
        self.fallback_reason = ""
        
        if CONFIG.ai_enabled:
            self.client = GeminiClient(CONFIG.gemini_api_key, CONFIG.retry, use_cache)
            self.available = self.client.available
            if not self.available:
                self.fallback_reason = self.client.error_msg
        else:
            if not CONFIG.gemini_api_key:
                self.fallback_reason = "GEMINI_API_KEY not found. Running in RULE_BASED mode."
            elif not GEMINI_AVAILABLE:
                self.fallback_reason = "google-genai not installed"
            else:
                self.fallback_reason = "Gemini disabled (ENABLE_GEMINI=false)"
    
    def _build_prompt(self, ticker: str, info: Dict, perf: Dict, events: List, news: List, val: Dict) -> str:
        """Build ONE comprehensive prompt"""
        # Performance
        perf_lines = []
        for w, d in perf.get('windows', {}).items():
            vol = d.get('vol_ann_pct')
            v_str = f", Vol={vol:.1f}%" if vol else ""
            perf_lines.append(f"  {w}D: {d['return_pct']:+.1f}%{v_str} (H=${d['high']:.2f}, L=${d['low']:.2f})")
        perf_text = "\n".join(perf_lines) or "  No data"
        
        # Events
        event_lines = []
        for e in events[:5]:
            matched = e.get('matched_news', [])
            if matched:
                title = matched[0]['title'][:55]
                event_lines.append(f"  {e['date_str']}: {e['type'].upper()} {e['return_pct']:+.1f}% → \"{title}...\"")
            else:
                event_lines.append(f"  {e['date_str']}: {e['type'].upper()} {e['return_pct']:+.1f}% → No clear news catalyst found")
        events_text = "\n".join(event_lines) or "  No significant events"
        
        # News
        news_lines = [f"  [{n['datetime_str']}] {n['title'][:65]}" for n in news[:8]]
        news_text = "\n".join(news_lines) or "  No recent news"
        
        # Fundamentals
        fund = []
        if info.get('pe_trailing'): fund.append(f"P/E={info['pe_trailing']:.1f}x")
        if info.get('pb_ratio'): fund.append(f"P/B={info['pb_ratio']:.1f}x")
        if info.get('roe'): fund.append(f"ROE={info['roe']*100:.1f}%")
        if info.get('profit_margin'): fund.append(f"Margin={info['profit_margin']*100:.1f}%")
        if info.get('revenue_growth'): fund.append(f"RevGrowth={info['revenue_growth']*100:.1f}%")
        if info.get('debt_to_equity'): fund.append(f"D/E={info['debt_to_equity']:.1f}")
        fund_text = ", ".join(fund) or "Limited data"
        
        # Valuation
        val_comps = [f"{c['method']}=${c['value']:.2f}" for c in val.get('components', [])]
        val_text = ", ".join(val_comps) or "N/A"
        
        prompt = f"""You are a senior equity analyst. Analyze {ticker} ({info.get('name', ticker)}).

=== MARKET DATA ===
Price: ${info.get('current_price', 0):.2f}
52W Range: ${info.get('fifty_two_week_low', 0):.2f} - ${info.get('fifty_two_week_high', 0):.2f}
Market Cap: ${(info.get('market_cap', 0) or 0)/1e9:.1f}B
Sector: {info.get('sector', 'Unknown')}

=== FUNDAMENTALS ===
{fund_text}

=== PERFORMANCE ===
{perf_text}

=== PRICE EVENTS (with matched news if found) ===
{events_text}

=== RECENT NEWS (reference only - do NOT hallucinate) ===
{news_text}

=== VALUATION MODEL OUTPUT ===
Fair Value: ${val.get('intrinsic_value', 0):.2f} (Range: ${val.get('range_low', 0):.2f}-${val.get('range_high', 0):.2f})
Components: {val_text}
Verdict: {val.get('verdict', 'N/A')} (Upside: {val.get('upside_pct', 0):+.1f}%)

=== INSTRUCTIONS ===
1. ONLY reference news headlines provided above
2. If no headline matches a price event, say "No clear news catalyst found"
3. Do NOT hallucinate or invent news
4. Return ONLY valid JSON in exact format below

=== REQUIRED JSON OUTPUT ===
{{
  "recommendation": "BUY" or "HOLD" or "SELL",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "summary": "2-3 sentence investment thesis",
  "price_event_analysis": "Explain biggest price move and link to news if available",
  "short_term_risks": ["risk1", "risk2"],
  "bull_case": ["point1", "point2", "point3"],
  "bear_case": ["point1", "point2", "point3"],
  "catalysts_to_watch": ["catalyst1", "catalyst2"],
  "trade_ideas": [
    {{"idea": "description", "entry": "$X", "stop": "$Y", "target": "$Z"}}
  ],
  "fair_value_low": {val.get('range_low', 0)},
  "fair_value_high": {val.get('range_high', 0)}
}}"""
        return prompt
    
    def analyze(self, ticker: str, info: Dict, perf: Dict, events: List, news: List, val: Dict, upcoming: Dict, debug: bool = False) -> Dict:
        """Main entry - makes ONE Gemini call (max)"""
        self.debug = debug or self.debug
        
        if not self.available:
            if self.debug:
                logger.info(f"[DEBUG] AI unavailable: {self.fallback_reason}")
            result = self._rule_based(ticker, info, perf, events, news, val)
            result['fallback_reason'] = self.fallback_reason
            return result
        
        prompt = self._build_prompt(ticker, info, perf, events, news, val)
        
        if self.debug:
            logger.info(f"[DEBUG] Prompt: {len(prompt)} chars, News: {len(news)}, Events: {len(events)}")
        
        # ONE Gemini call
        text = self.client.generate(prompt, ticker)
        
        if self.debug:
            s = self.client.stats
            logger.info(f"[DEBUG] API={s['api_calls']}, Cache={s['cache_hits']}, Retries={s['retries']}")
        
        if not text:
            result = self._rule_based(ticker, info, perf, events, news, val)
            result['fallback_reason'] = "Gemini API failed after retries"
            return result
        
        try:
            return self._parse(text, val)
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            result = self._rule_based(ticker, info, perf, events, news, val)
            result['fallback_reason'] = f"JSON parse error: {str(e)[:40]}"
            return result
    
    def _parse(self, text: str, val: Dict) -> Dict:
        """Parse Gemini response"""
        text = text.strip()
        if text.startswith('```'):
            text = '\n'.join(l for l in text.split('\n') if not l.startswith('```'))
        start = text.find('{')
        end = text.rfind('}') + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON found")
        result = json.loads(text[start:end])
        result.setdefault('recommendation', 'HOLD')
        result.setdefault('confidence', 'MEDIUM')
        result.setdefault('summary', 'Analysis complete.')
        result.setdefault('bull_case', [])
        result.setdefault('bear_case', [])
        low = result.get('fair_value_low', val.get('range_low', 0))
        high = result.get('fair_value_high', val.get('range_high', 0))
        result['fair_value'] = f"${low:.2f} - ${high:.2f}"
        result['source'] = 'GEMINI_AI'
        return result
    
    def _rule_based(self, ticker: str, info: Dict, perf: Dict, events: List, news: List, val: Dict) -> Dict:
        """Fallback rule-based analysis"""
        up = val.get('upside_pct', 0) or 0
        verdict = val.get('verdict', '')
        if up > 20: rec, conf = 'BUY', 'MEDIUM'
        elif up > 10: rec, conf = 'BUY', 'LOW'
        elif up > -10: rec, conf = 'HOLD', 'MEDIUM'
        elif up > -20: rec, conf = 'HOLD', 'LOW'
        else: rec, conf = 'SELL', 'LOW'
        
        perf_30d = perf.get('windows', {}).get(30, {}).get('return_pct', 0)
        name = info.get('name', ticker)
        summary = f"{name} appears {verdict.lower()}. Stock moved {perf_30d:+.1f}% over 30 days."
        if info.get('target_mean'):
            summary += f" Analysts target ${info['target_mean']:.2f}."
        
        # Event analysis
        event_analysis = "No significant price events detected."
        if events:
            e = events[0]
            matched = e.get('matched_news', [])
            if matched:
                event_analysis = f"Biggest move: {e['return_pct']:+.1f}% on {e['date_str']} linked to \"{matched[0]['title'][:45]}...\""
            else:
                event_analysis = f"Biggest move: {e['return_pct']:+.1f}% on {e['date_str']}. No clear news catalyst found."
        
        bull = []
        if info.get('revenue_growth') and info['revenue_growth'] > 0.10:
            bull.append(f"Strong revenue growth ({info['revenue_growth']*100:.1f}%)")
        if info.get('roe') and info['roe'] > 0.15:
            bull.append(f"High ROE ({info['roe']*100:.1f}%)")
        if up > 10:
            bull.append(f"Trading below fair value ({up:+.1f}%)")
        if not bull:
            bull = ['Established market position', 'Industry tailwinds']
        
        bear = []
        if info.get('pe_trailing') and info['pe_trailing'] > 30:
            bear.append(f"High valuation (P/E {info['pe_trailing']:.1f}x)")
        if up < -10:
            bear.append(f"Trading above fair value ({up:.1f}%)")
        if info.get('debt_to_equity') and info['debt_to_equity'] > 100:
            bear.append(f"High leverage (D/E {info['debt_to_equity']:.1f})")
        if not bear:
            bear = ['Competition risk', 'Macro uncertainty']
        
        return {
            'source': 'RULE_BASED',
            'recommendation': rec,
            'confidence': conf,
            'summary': summary,
            'price_event_analysis': event_analysis,
            'short_term_risks': ['Market volatility', 'Sector rotation'],
            'bull_case': bull[:3],
            'bear_case': bear[:3],
            'catalysts_to_watch': ['Earnings', 'Fed policy'],
            'trade_ideas': [],
            'fair_value': f"${val.get('range_low', 0):.2f} - ${val.get('range_high', 0):.2f}",
            'action': f"Fair value ${val.get('intrinsic_value', 0):.2f} vs current ${info.get('current_price', 0):.2f}. {verdict}."
        }

# =============================================================================
# REPORT
# =============================================================================

def generate_report(ticker: str, info: Dict, perf: Dict, events: List, news: List, val: Dict, analysis: Dict, upcoming: Dict, earnings: List) -> str:
    """Generate report"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'TRADING REPORTER v3.0':^80}")
    lines.append(f"{ticker} - {info.get('name', ticker)[:50]:^80}")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
    lines.append("")
    
    cp = val.get('current_price', 0) or 0
    iv = val.get('intrinsic_value', 0) or 0
    r_lo = val.get('range_low', 0) or 0
    r_hi = val.get('range_high', 0) or 0
    verdict = val.get('verdict', 'N/A')
    up = val.get('upside_pct', 0) or 0
    
    lines.append("-" * 80)
    lines.append(f"{ticker}: ${cp:.2f} → Fair ${iv:.2f} ({r_lo:.2f}-{r_hi:.2f}) | {verdict} | {up:+.1f}%")
    lines.append("-" * 80)
    lines.append("")
    
    rec = analysis.get('recommendation', 'N/A')
    conf = analysis.get('confidence', 'N/A')
    src = analysis.get('source', 'N/A')
    lines.append(f"RECOMMENDATION: {rec} (Confidence: {conf})")
    lines.append(f"Source: {src}")
    if analysis.get('fallback_reason'):
        lines.append(f"Note: {analysis['fallback_reason']}")
    lines.append("")
    lines.append(f"Summary: {analysis.get('summary', 'N/A')}")
    lines.append("")
    
    if analysis.get('price_event_analysis'):
        lines.append(f"Price Event: {analysis['price_event_analysis']}")
        lines.append("")
    
    lines.append("BULL CASE:")
    for b in analysis.get('bull_case', []):
        lines.append(f"  + {b}")
    lines.append("")
    lines.append("BEAR CASE:")
    for b in analysis.get('bear_case', []):
        lines.append(f"  - {b}")
    lines.append("")
    
    cats = analysis.get('catalysts_to_watch', [])
    if cats:
        lines.append("CATALYSTS:")
        for c in cats:
            lines.append(f"  • {c}")
        lines.append("")
    
    trades = analysis.get('trade_ideas', [])
    if trades:
        lines.append("TRADE IDEAS:")
        for t in trades:
            if isinstance(t, dict):
                lines.append(f"  • {t.get('idea', '')} | Entry: {t.get('entry', '')} | Stop: {t.get('stop', '')} | Target: {t.get('target', '')}")
            else:
                lines.append(f"  • {t}")
        lines.append("")
    
    lines.append("PERFORMANCE:")
    for w, d in perf.get('windows', {}).items():
        vol = d.get('vol_ann_pct')
        v_str = f" | Vol: {vol:.1f}%" if vol else ""
        lines.append(f"  {w}D: {d.get('return_pct', 0):+.1f}%{v_str} | H: ${d.get('high', 0):.2f} L: ${d.get('low', 0):.2f}")
    lines.append("")
    
    if events:
        lines.append("PRICE EVENTS:")
        for e in events[:5]:
            icon = "📈" if e['type'] == 'surge' else "📉"
            lines.append(f"  {icon} {e['date_str']}: {e['type'].upper()} {e['return_pct']:+.1f}%")
            for n in e.get('matched_news', [])[:1]:
                lines.append(f"      → {n['title'][:55]}...")
        lines.append("")
    
    if news:
        lines.append("RECENT NEWS:")
        for n in news[:5]:
            lines.append(f"  [{n['datetime_str']}] {n['title'][:55]}")
        lines.append("")
    
    lines.append("VALUATION:")
    lines.append(f"  Fair Value: {analysis.get('fair_value', 'N/A')}")
    for c in val.get('components', []):
        lines.append(f"  {c['method']}: ${c['value']:.2f} ({c['weight']*100:.0f}%)")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("DISCLAIMER: Educational only. NOT financial advice.")
    lines.append("=" * 80)
    return '\n'.join(lines)

# =============================================================================
# MAIN
# =============================================================================

def analyze_stock(ticker: str, no_ai: bool = False, quiet: bool = False, debug: bool = False, use_cache: bool = True) -> Dict[str, Any]:
    """Main entry - analyze stock with ONE Gemini call (max)"""
    ticker = ticker.upper()
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}...")
        print('='*60)
    
    if not quiet: print("[1/7] Fetching price data...")
    df = fetch_price_data(ticker)
    if df is None:
        print(f"  ✗ No price data for {ticker}")
        return {'ticker': ticker, 'error': 'No price data', 'file': None}
    if not quiet: print(f"  ✓ {len(df)} days")
    
    if not quiet: print("[2/7] Fetching fundamentals...")
    info = fetch_fundamentals(ticker)
    if not quiet: print(f"  ✓ {info.get('name', ticker)}")
    
    if not quiet: print("[3/7] Fetching news...")
    news = fetch_news(ticker, CONFIG.news_lookback_days)
    if not quiet: print(f"  ✓ {len(news)} articles")
    
    if not quiet: print("[4/7] Fetching events...")
    upcoming = fetch_upcoming_events(ticker)
    earnings = fetch_earnings_history(ticker)
    if not quiet: print(f"  ✓ {len(upcoming)} upcoming, {len(earnings)} earnings")
    
    if not quiet: print("[5/7] Analyzing performance...")
    perf = analyze_performance(df)
    events = detect_events(df)
    events = match_news_to_events(events, news)
    if not quiet: print(f"  ✓ {len(events)} price events")
    
    if not quiet: print("[6/7] Running valuation...")
    val = comprehensive_valuation(info)
    iv = val.get('intrinsic_value')
    if not quiet:
        if iv:
            print(f"  ✓ ${iv:.2f} ({val.get('verdict', 'N/A')})")
        else:
            print(f"  ⚠ Insufficient data")
    
    if not quiet: print("[7/7] AI analysis...")
    if no_ai:
        analyzer = AIAnalyzer(use_cache=False, debug=debug)
        analysis = analyzer._rule_based(ticker, info, perf, events, news, val)
        analysis['source'] = 'RULE_BASED (--no-ai)'
    else:
        analyzer = AIAnalyzer(use_cache=use_cache, debug=debug)
        analysis = analyzer.analyze(ticker, info, perf, events, news, val, upcoming, debug)
    
    if not quiet:
        src = analysis.get('source', 'Done')
        fb = analysis.get('fallback_reason', '')
        if fb:
            print(f"  ✓ {src} ({fb})")
        else:
            print(f"  ✓ {src}")
    
    report = generate_report(ticker, info, perf, events, news, val, analysis, upcoming, earnings)
    
    if not quiet:
        print("\n" + "=" * 80)
        cp = val.get('current_price', 0) or 0
        iv = val.get('intrinsic_value', 0) or 0
        print(f"{ticker}: ${cp:.2f} → Fair ${iv:.2f} | {val.get('verdict', 'N/A')} | {val.get('upside_pct', 0):+.1f}%")
        print(f"Recommendation: {analysis.get('recommendation', 'N/A')} ({analysis.get('confidence', 'N/A')})")
        print("=" * 80)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(CONFIG.output_dir) / f"{ticker}_REPORT_{ts}.txt"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    if not quiet:
        print(f"\n✓ Report saved: {filepath}")
    
    return {'ticker': ticker, 'file': str(filepath), 'valuation': val, 'analysis': analysis}

def main():
    parser = argparse.ArgumentParser(
        description='Trading Reporter v3.0 - Production Hardened',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python trading_reporter.py NVDA
    python trading_reporter.py AAPL MSFT GOOG --throttle 2.0
    python trading_reporter.py NVDA --no-ai
    python trading_reporter.py NVDA --debug
    python trading_reporter.py NVDA --no-cache
    python trading_reporter.py --clear-cache

Environment:
    GEMINI_API_KEY - API key from https://aistudio.google.com/app/apikey
        """
    )
    
    parser.add_argument('tickers', nargs='*', help='Stock ticker(s)')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI (rule-based only)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    parser.add_argument('--debug', action='store_true', help='Debug output (prompt size, cache, retries)')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    parser.add_argument('--throttle', type=float, default=CONFIG.default_throttle, help=f'Delay between tickers (default: {CONFIG.default_throttle}s)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached responses')
    
    args = parser.parse_args()
    
    if args.clear_cache:
        GeminiCache(CONFIG.cache_dir).clear()
        print("✓ Cache cleared")
        if not args.tickers:
            return 0
    
    if not args.tickers:
        print("\n" + "=" * 60)
        print("TRADING REPORTER v3.0 - Production Hardened")
        print("=" * 60)
        print("\nFeatures:")
        print("  ✓ Single Gemini call per stock (MAX 1)")
        print("  ✓ Exponential backoff for 429 errors")
        print("  ✓ Response caching")
        print("  ✓ Graceful fallback")
        print("\nInstall: pip install yfinance pandas numpy requests google-genai")
        print("Set: GEMINI_API_KEY=your_key")
        print("\nUsage: python trading_reporter.py NVDA")
        
        try:
            inp = input("\nEnter ticker(s): ").strip()
            args.tickers = inp.upper().split() if inp else []
        except (EOFError, KeyboardInterrupt):
            return 0
    
    if not args.tickers:
        print("No tickers")
        return 1
    
    use_cache = not args.no_cache
    
    for i, ticker in enumerate(args.tickers):
        try:
            analyze_stock(ticker.upper(), no_ai=args.no_ai, quiet=args.quiet, debug=args.debug, use_cache=use_cache)
            
            # Throttle between tickers
            if i < len(args.tickers) - 1 and args.throttle > 0:
                if not args.quiet:
                    print(f"\n⏳ Throttling {args.throttle}s...")
                time.sleep(args.throttle)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    print(f"\n✓ Done! Reports in {CONFIG.output_dir}/")
    return 0

if __name__ == '__main__':
    sys.exit(main())
