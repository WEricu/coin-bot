import requests
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 1. 系統日誌與環境變數
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TG_TOKEN = os.getenv("TG_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# 2. 監控配置 (ASI 是新幣，數據若不足會自動跳過)
COINS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "XRP-USDT-SWAP", "BNB-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP", "ADA-USDT-SWAP", "TRX-USDT-SWAP", "AVAX-USDT-SWAP", "SUI-USDT-SWAP"]
LOG_FILE = "active_trades.csv"
HISTORY_FILE = "trade_history.csv"

def send_tg(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        res.raise_for_status()
    except: pass

def fetch_okx(instId, bar="15m", limit="300"):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={instId}&bar={bar}&limit={limit}"
        res = requests.get(url, timeout=10).json()
        if 'data' not in res or not res['data']: return None
        df = pd.DataFrame(res['data'], columns=['ts', 'o', 'h', 'l', 'c', 'v', 'volCcy', 'volCcyQuote', 'confirm'])
        df[['o', 'h', 'l', 'c', 'v']] = df[['o', 'h', 'l', 'c', 'v']].astype(float)
        df = df[df['confirm'] == "1"].copy()
        return df.iloc[::-1].reset_index(drop=True)
    except: return None

def get_sentiment(instId):
    try:
        ls_res = requests.get(f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?instId={instId}&period=5m").json()
        ls_curr, ls_prev = float(ls_res['data'][0][1]), float(ls_res['data'][2][1])
        base = instId.split('-')[0]
        s_df = fetch_okx(f"{base}-USDT", bar="5m", limit="20")
        cvd_up = s_df['c'].iloc[-1] > s_df['c'].iloc[-10] if s_df is not None else False
        oi_res = requests.get(f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?instId={instId}&period=5m").json()
        fuel = float(oi_res['data'][0][1]) < float(oi_res['data'][2][1]) if len(oi_res.get('data', [])) > 2 else False
        return ls_curr, ls_prev, cvd_up, fuel
    except: return 1.0, 1.0, False, False

def main():
    now_tw = datetime.utcnow() + timedelta(hours=8)
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["instId", "side", "entry", "sl", "tp1", "tp3", "tp1_hit"]).to_csv(LOG_FILE, index=False)
    
    trades = pd.read_csv(LOG_FILE).to_dict('records')
    still_active, finished = [], []

    # 監控持倉
    for t in trades:
        df = fetch_okx(t['instId'], "15m", "10")
        if df is None or df.empty: still_active.append(t); continue
        curr_p, hi, lo = df['c'].iloc[-1], df['h'].max(), df['l'].min()
        
        if (t['side'] == "LONG" and lo <= t['sl']) or (t['side'] == "SHORT" and hi >= t['sl']):
            send_tg(f"❌ *止損離場*\n💰 #{t['instId']} | 價格: `{curr_p}`"); finished.append(t); continue

        if t.get('tp1_hit', 0) == 0:
            if (t['side'] == "LONG" and hi >= t['tp1']) or (t['side'] == "SHORT" and lo <= t['tp1']):
                t['tp1_hit'] = 1; t['sl'] = t['entry'] 
                send_tg(f"🔹 *TP1 達成：自動保本*\n💰 #{t['instId']} | 止損移至: `{t['sl']}`")
        
        if (t['side'] == "LONG" and hi >= t['tp3']) or (t['side'] == "SHORT" and lo <= t['tp3']):
            send_tg(f"🚀 *TP3 終極止盈！*\n💰 #{t['instId']}"); finished.append(t)
        else:
            still_active.append(t)

    # 掃描新訊號
    for instId in COINS:
        if instId in [x['instId'] for x in still_active]: continue
        df_4h = fetch_okx(instId, "4H", "300")
        if df_4h is None or len(df_4h) < 200: continue
        
        ema200 = df_4h['c'].ewm(span=200, adjust=False).mean().iloc[-1]
        ls_c, ls_p, cvd_up, fuel = get_sentiment(instId)
        df_15 = fetch_okx(instId, "15m", "100")
        if df_15 is None: continue
        
        curr_p = df_15['c'].iloc[-1]
        atr = (df_15['h'] - df_15['l']).rolling(14).mean().iloc[-1]
        h_max, l_min = df_15['h'].iloc[-20:-2].max(), df_15['l'].iloc[-20:-2].min()

        long_c = (curr_p > ema200) and (curr_p > h_max) and cvd_up and (ls_c < ls_p) and fuel
        short_c = (curr_p < ema200) and (curr_p < l_min) and (not cvd_up) and (ls_c > ls_p) and fuel

        if long_c or short_c:
            side = "LONG" if long_c else "SHORT"
            sl = curr_p - (atr * 1.5) if long_c else curr_p + (atr * 1.5)
            tp1, tp3 = curr_p + atr if long_c else curr_p - atr, curr_p + atr*4 if long_c else curr_p - atr*4
            still_active.append({"instId": instId, "side": side, "entry": curr_p, "sl": sl, "tp1": tp1, "tp3": tp3, "tp1_hit": 0})
            send_tg(f"🎯 *Alpha 燃料狙擊*\n💎 #{instId.split('-')[0]} | {side}\n📍 進場: `{curr_p:.4f}`\n🚫 止損: `{sl:.4f}`\n⛽ 燃料: `🔥 點燃`")

    pd.DataFrame(still_active).to_csv(LOG_FILE, index=False)
    if finished: pd.DataFrame(finished).to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)

if __name__ == "__main__":
    main()
