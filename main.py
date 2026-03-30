import requests
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TG_TOKEN = os.getenv('TG_TOKEN')
CHAT_ID  = os.getenv('CHAT_ID')

COINS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "XRP-USDT-SWAP", "BNB-USDT-SWAP", "SOL-USDT-SWAP",
         "DOGE-USDT-SWAP", "ADA-USDT-SWAP", "TRX-USDT-SWAP", "AVAX-USDT-SWAP", "SUI-USDT-SWAP",
         "PENGU-USDT-SWAP", "RIVER-USDT-SWAP", "AAVE-USDT-SWAP"]

LOG_FILE     = "active_trades.csv"
HISTORY_FILE = "trade_history.csv"
def send_tg(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=15)
        res.raise_for_status()
        logging.info('TG sent OK')
    except Exception as e:
        logging.error(f'TG error: {e}')


def fetch_okx(instId, bar='15m', limit='300'):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={instId}&bar={bar}&limit={limit}"
        res = requests.get(url, timeout=10).json()
        if 'data' not in res or not res['data']:
            logging.warning(f'No data for {instId} bar={bar}')
            return None
        df = pd.DataFrame(res['data'], columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
        df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
        df = df[df['confirm'] == '1'].copy()
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        logging.error(f'fetch_okx error {instId}: {e}')
        return None
def get_sentiment(instId):
    """Returns (ls_curr, ls_prev, cvd_up, ls_ok).
    ls_ok=False means ratio data unavailable -- callers should skip the ls filter."""
    ls_curr, ls_prev, ls_ok = 1.0, 1.0, False
    cvd_up = False
    try:
        ls_res = requests.get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
            f"?instId={instId}&period=5m", timeout=10).json()
        ls_data = ls_res.get('data', [])
        if len(ls_data) >= 2:
            ls_curr = float(ls_data[0][1])
            ls_prev = float(ls_data[-1][1])
            ls_ok   = True
            logging.info(f'  ls {instId}: curr={ls_curr:.3f} prev={ls_prev:.3f}')
        else:
            logging.warning(f'  ls unavailable {instId} ({len(ls_data)} rows) -- skipping ls filter')
    except Exception as e:
        logging.error(f'get_sentiment ls error {instId}: {e}')
    try:
        s_df   = fetch_okx(instId, bar='5m', limit='20')
        cvd_up = bool(s_df is not None and len(s_df) >= 10 and s_df['c'].iloc[-1] > s_df['c'].iloc[-10])
    except Exception as e:
        logging.error(f'get_sentiment cvd error {instId}: {e}')
    return ls_curr, ls_prev, cvd_up, ls_ok
def main():
    now_tw = datetime.utcnow() + timedelta(hours=8)
    logging.info(f'=== Alpha Oracle start {now_tw.strftime("%Y-%m-%d %H:%M")} TW, {len(COINS)} coins ===')

    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['instId','side','entry','sl','tp1','tp3','tp1_hit']).to_csv(LOG_FILE, index=False)

    trades = pd.read_csv(LOG_FILE).to_dict('records')
    still_active, finished = [], []

    for t in trades:
        df = fetch_okx(t['instId'], '15m', '10')
        if df is None or df.empty:
            still_active.append(t); continue
        curr_p, hi, lo = df['c'].iloc[-1], df['h'].max(), df['l'].min()
        if (t['side'] == 'LONG' and lo <= t['sl']) or (t['side'] == 'SHORT' and hi >= t['sl']):
            send_tg(f"\u274c *\u6b62\u640d\u96e2\u5834*\n#{t['instId']} | {curr_p}")
            finished.append(t); continue
        if t.get('tp1_hit', 0) == 0:
            if (t['side'] == 'LONG' and hi >= t['tp1']) or (t['side'] == 'SHORT' and lo <= t['tp1']):
                t['tp1_hit'] = 1; t['sl'] = t['entry']
                send_tg(f"\U0001f539 *TP1 \u9054\u6210\uff1a\u4fdd\u672c*\n#{t['instId']}")
        if (t['side'] == 'LONG' and hi >= t['tp3']) or (t['side'] == 'SHORT' and lo <= t['tp3']):
            send_tg(f"\U0001f680 *TP3 \u6b62\u76c8*\n#{t['instId']}")
            finished.append(t)
        else:
            still_active.append(t)
    for instId in COINS:
        if instId in [x['instId'] for x in still_active]:
            continue
        logging.info(f'[SCAN] {instId}')
        df_4h = fetch_okx(instId, '4H', '300')
        if df_4h is None or len(df_4h) < 50:
            logging.warning(f'[SKIP] {instId} insufficient 4H data')
            continue
        span   = min(200, len(df_4h))
        ema200 = df_4h['c'].ewm(span=span, adjust=False).mean().iloc[-1]

        ls_c, ls_p, cvd_up, ls_ok = get_sentiment(instId)

        df_15 = fetch_okx(instId, '15m', '100')
        if df_15 is None or len(df_15) < 25:
            continue
        curr_p = df_15['c'].iloc[-1]
        atr    = (df_15['h'] - df_15['l']).rolling(14).mean().iloc[-1]
        h_max  = df_15['h'].iloc[-20:-2].max()
        l_min  = df_15['l'].iloc[-20:-2].min()

        # ls filter: if no data, do not block the signal
        ls_long_ok  = (not ls_ok) or (ls_c < ls_p)
        ls_short_ok = (not ls_ok) or (ls_c > ls_p)

        long_c  = (curr_p > ema200) and (curr_p > h_max) and cvd_up       and ls_long_ok
        short_c = (curr_p < ema200) and (curr_p < l_min) and (not cvd_up) and ls_short_ok

        logging.info(
            f'  {instId} p={curr_p:.4f} ema={ema200:.4f} ls_ok={ls_ok} '
            f'LONG={long_c}(p>ema:{curr_p>ema200},brkout:{curr_p>h_max},cvd:{cvd_up},ls:{ls_long_ok}) '
            f'SHORT={short_c}(p<ema:{curr_p<ema200},brkdn:{curr_p<l_min},!cvd:{not cvd_up},ls:{ls_short_ok})'
        )
        if long_c or short_c:
            side = 'LONG' if long_c else 'SHORT'
            sl   = curr_p - (atr * 1.5) if long_c else curr_p + (atr * 1.5)
            tp1  = curr_p + atr          if long_c else curr_p - atr
            tp3  = curr_p + atr * 4      if long_c else curr_p - atr * 4
            still_active.append({'instId': instId, 'side': side, 'entry': curr_p,
                                  'sl': sl, 'tp1': tp1, 'tp3': tp3, 'tp1_hit': 0})
            coin = instId.split('-')[0]
            send_tg(
                f"\U0001f3af *Alpha \u72d9\u64ca\u8a0a\u865f*\n"
                f"\U0001f48e #{coin} | {side}\n"
                f"\U0001f4cd \u9032\u5834: {curr_p:.4f}\n"
                f"\U0001f6ab \u6b62\u640d: {sl:.4f}\n"
                f"\U0001f3c6 TP1: {tp1:.4f} | TP3: {tp3:.4f}"
            )

    pd.DataFrame(still_active).to_csv(LOG_FILE, index=False)
    if finished:
        pd.DataFrame(finished).to_csv(HISTORY_FILE, mode='a',
                                      header=not os.path.exists(HISTORY_FILE), index=False)
    logging.info(f'=== Done, active: {len(still_active)} ===')


if __name__ == '__main__':
    main()
