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

def detect_smc(df_1h):
    """BOS + Order Block on 1H. Returns (bias, ob_high, ob_low, info)"""
    if df_1h is None or len(df_1h) < 20:
        return 'NEUTRAL', None, None, 'no data'
    try:
        df = df_1h.iloc[-30:].reset_index(drop=True)
        def find_pivots(series, side, w=2):
            pts = []
            for i in range(w, len(series) - w):
                v = series.iloc[i]
                nb = [series.iloc[i+k] for k in range(-w, w+1) if k != 0]
                if side == 'h' and all(v >= n for n in nb): pts.append((i, v))
                elif side == 'l' and all(v <= n for n in nb): pts.append((i, v))
            return pts
        ph = find_pivots(df['h'], 'h')
        pl = find_pivots(df['l'], 'l')
        if len(ph) < 2 or len(pl) < 2:
            return 'NEUTRAL', None, None, f'pivots h={len(ph)} l={len(pl)}'
        (ph1_i, ph1), (_, ph2) = ph[-1], ph[-2]
        (pl1_i, pl1), (_, pl2) = pl[-1], pl[-2]
        hh = ph1 > ph2; hl = pl1 > pl2
        ll = pl1 < pl2; lh = ph1 < ph2
        info = f'HH={hh} HL={hl} LL={ll} LH={lh} ph={ph1:.4f} pl={pl1:.4f}'
        if hh and hl:
            bias = 'BULL'
            ob_idx = next((i for i in range(ph1_i-1, max(0,ph1_i-12)-1, -1)
                           if df['o'].iloc[i] > df['c'].iloc[i]), None)
            ob_high = df['h'].iloc[ob_idx] if ob_idx is not None else pl1*1.005
            ob_low  = df['l'].iloc[ob_idx] if ob_idx is not None else pl1*0.995
        elif ll and lh:
            bias = 'BEAR'
            ob_idx = next((i for i in range(pl1_i-1, max(0,pl1_i-12)-1, -1)
                           if df['c'].iloc[i] > df['o'].iloc[i]), None)
            ob_high = df['h'].iloc[ob_idx] if ob_idx is not None else ph1*1.005
            ob_low  = df['l'].iloc[ob_idx] if ob_idx is not None else ph1*0.995
        else:
            return 'NEUTRAL', None, None, info
        return bias, ob_high, ob_low, info
    except Exception as e:
        logging.error(f'detect_smc error: {e}')
        return 'NEUTRAL', None, None, str(e)


def get_sentiment(instId):
    ls_curr, ls_prev, ls_ok = 1.0, 1.0, False
    cvd_up = False
    try:
        ls_res = requests.get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
            f"?instId={instId}&period=5m", timeout=10).json()
        ls_data = ls_res.get('data', [])
        if len(ls_data) >= 2:
            ls_curr = float(ls_data[0][1]); ls_prev = float(ls_data[-1][1]); ls_ok = True
            logging.info(f'  ls {instId}: {ls_curr:.3f} vs {ls_prev:.3f}')
        else:
            logging.warning(f'  ls unavailable {instId}')
    except Exception as e:
        logging.error(f'ls error {instId}: {e}')
    try:
        s_df = fetch_okx(instId, bar='5m', limit='20')
        cvd_up = bool(s_df is not None and len(s_df) >= 10 and s_df['c'].iloc[-1] > s_df['c'].iloc[-10])
    except Exception as e:
        logging.error(f'cvd error {instId}: {e}')
    return ls_curr, ls_prev, cvd_up, ls_ok

def main():
    now_tw = datetime.utcnow() + timedelta(hours=8)
    logging.info(f'=== Alpha Oracle SMC {now_tw.strftime("%Y-%m-%d %H:%M")} TW {len(COINS)} coins ===')
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['instId','side','entry','sl','tp1','tp3','tp1_hit']).to_csv(LOG_FILE, index=False)
    trades = pd.read_csv(LOG_FILE).to_dict('records')
    still_active, finished = [], []
    for t in trades:
        df = fetch_okx(t['instId'], '15m', '10')
        if df is None or df.empty: still_active.append(t); continue
        curr_p, hi, lo = df['c'].iloc[-1], df['h'].max(), df['l'].min()
        if (t['side']=='LONG' and lo<=t['sl']) or (t['side']=='SHORT' and hi>=t['sl']):
            send_tg(f"\u274c *\u6b62\u640d* #{t['instId']} {t['side']} @ {curr_p:.4f}")
            finished.append(t); continue
        if t.get('tp1_hit',0)==0:
            if (t['side']=='LONG' and hi>=t['tp1']) or (t['side']=='SHORT' and lo<=t['tp1']):
                t['tp1_hit']=1; t['sl']=t['entry']
                send_tg(f"\U0001f539 *TP1 \u9054\u6210 \u4fdd\u672c* #{t['instId']}")
        if (t['side']=='LONG' and hi>=t['tp3']) or (t['side']=='SHORT' and lo<=t['tp3']):
            send_tg(f"\U0001f680 *TP3 \u6b62\u76c8* #{t['instId']}")
            finished.append(t)
        else:
            still_active.append(t)
    for instId in COINS:
        if instId in [x['instId'] for x in still_active]: continue
        logging.info(f'[SCAN] {instId}')
        df_4h = fetch_okx(instId, '4H', '200')
        df_1h = fetch_okx(instId, '1H', '50')
        df_15 = fetch_okx(instId, '15m', '100')
        if df_4h is None or len(df_4h)<50 or df_15 is None or len(df_15)<25:
            logging.warning(f'[SKIP] {instId}'); continue
        span   = min(200, len(df_4h))
        ema200 = df_4h['c'].ewm(span=span, adjust=False).mean().iloc[-1]
        curr_p = df_15['c'].iloc[-1]
        atr    = (df_15['h'] - df_15['l']).rolling(14).mean().iloc[-1]
        bias, ob_high, ob_low, smc_info = detect_smc(df_1h)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
TG_TOKEN = os.getenv('TG_TOKEN')
CHAT_ID  = os.getenv('CHAT_ID')
COINS = ["BTC-USDT-SWAP","ETH-USDT-SWAP","XRP-USDT-SWAP","BNB-USDT-SWAP","SOL-USDT-SWAP",
         "DOGE-USDT-SWAP","ADA-USDT-SWAP","TRX-USDT-SWAP","AVAX-USDT-SWAP","SUI-USDT-SWAP",
         "PENGU-USDT-SWAP","RIVER-USDT-SWAP","AAVE-USDT-SWAP"]
LOG_FILE = "active_trades.csv"
HIST_FILE = "trade_history.csv"

def send_tg(msg):
    try:
        r = requests.post(f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id':CHAT_ID,'text':msg,'parse_mode':'Markdown'}, timeout=15)
        r.raise_for_status(); logging.info('TG OK')
    except Exception as e: logging.error(f'TG error {e}')

def fetch_okx(instId, bar='15m', limit='300'):
    try:
        res = requests.get(f'https://www.okx.com/api/v5/market/candles?instId={instId}&bar={bar}&limit={limit}',timeout=10).json()
        if not res.get('data'): return None
        df = pd.DataFrame(res['data'],columns=['ts','o','h','l','c','v','vq','vb','confirm'])
        df[['o','h','l','c','v']] = df[['o','h','l','c','v']].astype(float)
        return df[df['confirm']=='1'].iloc[::-1].reset_index(drop=True)
    except Exception as e: logging.error(f'fetch_okx {instId} {e}'); return None

def detect_smc(df_1h):
    if df_1h is None or len(df_1h) < 20: return 'NEUTRAL',None,None,'no data'
    try:
        df = df_1h.iloc[-30:].reset_index(drop=True)
        def pivots(s, side, w=2):
            out = []
            for i in range(w, len(s)-w):
                nb = [s.iloc[i+k] for k in range(-w,w+1) if k]
                if side=='h' and all(s.iloc[i]>=n for n in nb): out.append((i,s.iloc[i]))
                if side=='l' and all(s.iloc[i]<=n for n in nb): out.append((i,s.iloc[i]))
            return out
        ph = pivots(df['h'],'h'); pl = pivots(df['l'],'l')
        if len(ph)<2 or len(pl)<2: return 'NEUTRAL',None,None,f'ph={len(ph)} pl={len(pl)}'
        (ph1i,ph1),(_,ph2) = ph[-1],ph[-2]
        (pl1i,pl1),(_,pl2) = pl[-1],pl[-2]
        hh=ph1>ph2; hl=pl1>pl2; ll=pl1<pl2; lh=ph1<ph2
        info=f'HH={hh} HL={hl} LL={ll} LH={lh} ph={ph1:.4f} pl={pl1:.4f}'
        if hh and hl:
            bias='BULL'
            i = next((i for i in range(ph1i-1,max(0,ph1i-12)-1,-1) if df['o'].iloc[i]>df['c'].iloc[i]),None)
            ob_hi = df['h'].iloc[i] if i is not None else pl1*1.005
            ob_lo = df['l'].iloc[i] if i is not None else pl1*0.995
        elif ll and lh:
            bias='BEAR'
            i = next((i for i in range(pl1i-1,max(0,pl1i-12)-1,-1) if df['c'].iloc[i]>df['o'].iloc[i]),None)
            ob_hi = df['h'].iloc[i] if i is not None else ph1*1.005
            ob_lo = df['l'].iloc[i] if i is not None else ph1*0.995
        else: return 'NEUTRAL',None,None,info
        return bias,ob_hi,ob_lo,info
    except Exception as e: logging.error(f'smc error {e}'); return 'NEUTRAL',None,None,str(e)

def get_sentiment(instId):
    lc,lp,lok = 1.0,1.0,False; cvd=False
    try:
        d = requests.get(f'https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId={instId}&period=5m',timeout=10).json().get('data',[])
        if len(d)>=2: lc=float(d[0][1]); lp=float(d[-1][1]); lok=True; logging.info(f'  ls {lc:.3f}v{lp:.3f}')
        else: logging.warning(f'  ls N/A {instId}')
    except Exception as e: logging.error(f'ls {e}')
    try:
        s = fetch_okx(instId,'5m','20')
        cvd = bool(s is not None and len(s)>=10 and s['c'].iloc[-1]>s['c'].iloc[-10])
    except Exception as e: logging.error(f'cvd {e}')
    return lc,lp,cvd,lok

def main():
    now = datetime.utcnow()+timedelta(hours=8)
    logging.info(f'=== SMC Bot {now.strftime("%m-%d %H:%M")} TW {len(COINS)} coins ===')
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['instId','side','entry','sl','tp1','tp3','tp1_hit']).to_csv(LOG_FILE,index=False)
    trades = pd.read_csv(LOG_FILE).to_dict('records')
    active,done = [],[]
    for t in trades:
        df = fetch_okx(t['instId'],'15m','10')
        if df is None or df.empty: active.append(t); continue
        p,hi,lo = df['c'].iloc[-1],df['h'].max(),df['l'].min()
        if (t['side']=='LONG' and lo<=t['sl']) or (t['side']=='SHORT' and hi>=t['sl']):
            send_tg(f"SL #{t['instId']} {t['side']} {p:.4f}"); done.append(t); continue
        if t.get('tp1_hit',0)==0:
            if (t['side']=='LONG' and hi>=t['tp1']) or (t['side']=='SHORT' and lo<=t['tp1']):
                t['tp1_hit']=1; t['sl']=t['entry']
                send_tg(f"TP1 done - breakeven #{t['instId']}")
        if (t['side']=='LONG' and hi>=t['tp3']) or (t['side']=='SHORT' and lo<=t['tp3']):
            send_tg(f"TP3 hit #{t['instId']}"); done.append(t)
        else: active.append(t)
    for instId in COINS:
        if instId in [x['instId'] for x in active]: continue
        logging.info(f'[SCAN] {instId}')
        df4 = fetch_okx(instId,'4H','200')
        df1 = fetch_okx(instId,'1H','50')
        df15= fetch_okx(instId,'15m','100')
        if df4 is None or len(df4)<50 or df15 is None or len(df15)<25:
            logging.warning(f'skip {instId}'); continue
        ema200 = df4['c'].ewm(span=min(200,len(df4)),adjust=False).mean().iloc[-1]
        p   = df15['c'].iloc[-1]
        atr = (df15['h']-df15['l']).rolling(14).mean().iloc[-1]
        bias,ob_hi,ob_lo,sinfo = detect_smc(df1)
        lc,lp,cvd,lok = get_sentiment(instId)
        ll_ok = (not lok) or (lc<lp)
        ls_ok = (not lok) or (lc>lp)
        at_ob = ob_hi is not None and ob_lo*0.998<=p<=ob_hi*1.002
        long_c  = bias=='BULL' and at_ob and p>=ema200*0.98 and cvd     and ll_ok
        short_c = bias=='BEAR' and at_ob and p<=ema200*1.02 and not cvd and ls_ok
        obl = f'{ob_lo:.4f}' if ob_lo else 'N/A'
        obh = f'{ob_hi:.4f}' if ob_hi else 'N/A'
        logging.info(f'  {instId} p={p:.4f} ema={ema200:.4f} bias={bias} OB=[{obl}-{obh}] at_ob={at_ob} cvd={cvd}')
        logging.info(f'  LONG={long_c} SHORT={short_c} | {sinfo}')
        if long_c or short_c:
            side='LONG' if long_c else 'SHORT'
            sl  = p-atr*1.5 if long_c else p+atr*1.5
            tp1 = p+atr     if long_c else p-atr
            tp3 = p+atr*4   if long_c else p-atr*4
            active.append({'instId':instId,'side':side,'entry':p,'sl':sl,'tp1':tp1,'tp3':tp3,'tp1_hit':0})
            coin=instId.split('-')[0]
            send_tg(f'SMC Signal\n{coin} {side}\nEntry:{p:.4f} SL:{sl:.4f}\nTP1:{tp1:.4f} TP3:{tp3:.4f}\nOB:{obl}-{obh} {bias}')
    pd.DataFrame(active).to_csv(LOG_FILE,index=False)
    if done: pd.DataFrame(done).to_csv(HIST_FILE,mode='a',header=not os.path.exists(HIST_FILE),index=False)
    logging.info(f'=== Done active={len(active)} ===')

if __name__=='__main__': main()
