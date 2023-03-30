import pandas as pd
from binance.client import Client
import numpy as np

def calc_rsi(df, column="Close", period=14):
    delta = df[column].diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    rUp = up.ewm(com=period - 1, adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()

    rsi = 100 - 100 / (1 + rUp / rDown)

    return df.join(rsi.to_frame(f'{column}-RSI'))


def get_binance_close(ticker, interval='1h', limit=100, market='perp', client=None):
    
    if client == None:
        client = Client("", "")
    
    columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'QAV', 'Num Trades', 'BBAV', 'BQAV', 'Misc']
    output_columns = ['Close', 'Volume']
    
    if market.upper() == 'PERP':
        targ_url = 'https://fapi.binance.com/fapi/v1/klines'
    elif market.upper() == 'SPOT':
        targ_url = 'https://api.binance.com/api/v3/klines'

    params = {
            'symbol': ticker,
            'interval': interval,
            'limit': limit,
    }

    output_data = client._request('get', targ_url, signed=False, data=params)
    df = pd.DataFrame(output_data, columns=columns)
    
    # Converting UNIX stamps to datetimes
    df['Close Time'] = df['Close Time'] / 1000
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='s')
    df['Close Time'] = df['Close Time'].dt.round('1min')

    df = df.set_index('Close Time')
    df = df[output_columns]
    df = df.astype(float)
    df['Volume'] = df['Volume'] * df['Close'] / 1000000 # mio

    return df


def get_binance_perp_basis(ticker, interval='1h', limit=100):
    
    spot = get_binance_close(ticker, interval=interval, limit=limit, market='spot')
    perp = get_binance_close(ticker, interval=interval, limit=limit, market='perp')
    
    df = spot
    df.rename(columns = {'Close': 'Spot'}, inplace = True)
    df['Perp'] = perp['Close']
    df['Basis'] = 100 * ((df['Perp'] - df['Spot']) / df['Spot'])
    
    return df


def get_binance_open_interest(ticker, interval='1h', limit=100, client=None):
    
    if client == None:
        client = Client("", "")
    
    columns = ['Ticker', 'Base OI', 'OI', 'Close Time']
    output_columns = ['OI']
    
    targ_url = 'https://fapi.binance.com/futures/data/openInterestHist'
    
    params = {
            'symbol': ticker,
            'period': interval,
            'limit': limit,
    }

    output_data = client._request('get', targ_url, signed=False, data=params)
    df = pd.DataFrame(output_data)
    
     # Converting UNIX stamps to datetimes
    df['timestamp'] = df['timestamp'] / 1000
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = df['timestamp'].dt.round('1min')
    df = df.set_index('timestamp')
    df.rename(columns = {'sumOpenInterestValue': 'OI'}, inplace = True)
    
    df = pd.DataFrame(df['OI'], columns=['OI'])
    df = df.astype(float)
    df['OI'] = df['OI'] / 1000000 # mm
    
    return df


def get_binance_taker_volume(ticker, interval='1h', limit=100, client=None):
    
    if client == None:
        client = Client("", "")
    
    # Futures data
    
    targ_url = 'https://fapi.binance.com/futures/data/takerlongshortRatio'
    
    params = {
            'symbol': ticker,
            'period': interval,
            'limit': limit,
    }

    output_data = client._request('get', targ_url, signed=False, data=params)
    df = pd.DataFrame(output_data)
    
    # Converting UNIX stamps to datetimes
    df['timestamp'] = df['timestamp'] / 1000
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = df['timestamp'].dt.round('1min')
    df = df.set_index('timestamp')
    
    df = df[['buyVol', 'sellVol']]
    df.rename(columns={'buyVol': 'Perp Buy Vol', 'sellVol': 'Perp Sell Vol'}, inplace=True)
    df = df.astype(float)
    
    # df['Perp CVD'] = (df['Perp Buy Vol'] - df['Perp Sell Vol']) / (df['Perp Buy Vol'] + df['Perp Sell Vol'])
    df['Perp TBSR'] = df['Perp Buy Vol'] / df['Perp Sell Vol']
    
    # Spot data
    
    columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'timestamp', 'QAV', 'Num Trades', 'BBAV', 'BQAV', 'Misc']
    output_columns = ['Volume', 'BBAV']
    
    targ_url = 'https://api.binance.com/api/v3/klines'
    
    params = {
        'symbol': ticker,
        'interval': interval,
        'limit': limit,
    }

    output_data = client._request('get', targ_url, signed=False, data=params)
    df_spot = pd.DataFrame(output_data, columns=columns)
    
    # Converting UNIX stamps to datetimes
    df_spot['timestamp'] = df_spot['timestamp'] / 1000
    df_spot['timestamp'] = pd.to_datetime(df_spot['timestamp'], unit='s')
    df_spot['timestamp'] = df_spot['timestamp'].dt.round('1min')
    df_spot = df_spot.set_index('timestamp')
    df_spot = df_spot[output_columns]
    df_spot = df_spot.astype(float)
    
    df_spot['Spot Sell Vol'] = df_spot['Volume'] - df_spot['BBAV']
    df_spot.rename(columns = {'BBAV': 'Spot Buy Vol'}, inplace = True)
    df_spot = df_spot[['Spot Buy Vol', 'Spot Sell Vol']]
    
    # df_spot['Spot CVD'] = (df_spot['Spot Buy Vol'] - df_spot['Spot Sell Vol']) / (df_spot['Spot Buy Vol'] + df_spot['Spot Sell Vol'])
    df_spot['Spot TBSR'] = df_spot['Spot Buy Vol'] / df_spot['Spot Sell Vol']
    
    df_merged = pd.merge(df, df_spot, left_index=True, right_index=True)
    
    return df_merged


def get_binance_funding(ticker, limit=100, client=None):
    
    if client == None:
        client = Client("", "")
    
    # Historical funding rates
    
    params = {
            'symbol': ticker,
            'limit': limit,
    }
    
    
    targ_url = 'https://fapi.binance.com/fapi/v1/fundingRate'
    # targ_url = 'https://fapi.binance.com/fapi/v1/premiumIndex' -> USED TO GET LIVE FUNDING

    output_data = client._request('get', targ_url, signed=False, data=params)
    df = pd.DataFrame(output_data)
    
    # Converting UNIX stamps to datetimes
    df['Close Time'] = df['fundingTime']
    df['Close Time'] = df['Close Time'] / 1000
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='s')
    df['Close Time'] = df['Close Time'].dt.round('1min')

    df = df.set_index('Close Time')
    df = df['fundingRate']
    df = df.astype(float)

    df = df * 100
    
    # Live funding rate
    
    params = {
            'symbol': ticker,
    }
    
    targ_url = 'https://fapi.binance.com/fapi/v1/premiumIndex'

    output_data = client._request('get', targ_url, signed=False, data=params)
    output_data['nextFundingTime'] = pd.to_datetime(output_data['nextFundingTime'] / 1000, unit='s')
    
    row = pd.Series({output_data['nextFundingTime']: float(output_data['lastFundingRate']) * 100})
    df = df.append(row)
    df = pd.DataFrame(df, columns=['Funding'])

    return df


def get_binance_long_short_ratio(ticker, interval='1h', limit=100, account_type='global', client=None):

    if client == None:
        client = Client("", "")
        
    if account_type == 'global':
        targ_url = 'https://fapi.binance.com/futures/data/globalLongShortAccountRatio'
    else:
        targ_url = 'https://fapi.binance.com/futures/data/topLongShortAccountRatio'

    params = {
        'symbol': ticker,
        'period': interval,
        'limit': limit
    }

    output_data = client._request('get', targ_url, signed=False, data=params)
    df = pd.DataFrame(output_data)

    # Converting UNIX stamps to datetimes
    df['Close time'] = df['timestamp']
    df['Close time'] = df['Close time'] / 1000
    df['Close time'] = pd.to_datetime(df['Close time'], unit='s')
    df['Close time'] = df['Close time'].dt.round('1min')

    df = df.set_index('Close time')
    df = df['longShortRatio']
    df = df.astype(float)

    candles = client.futures_klines(symbol=ticker, interval=interval, limit=limit)
    df_input = pd.DataFrame(candles)
    df_input.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'QAV', 'NoT', 'TbbAV', 'TBQAV', 'x']

    df_input['Close time'] = df_input['Close time'] / 1000
    df_input['Close time'] = pd.to_datetime(df_input['Close time'], unit='s')
    df_input['Close time'] = df_input['Close time'].dt.round('1min')

    df_input = df_input.set_index('Close time')
    df_input = df_input[['Close']]
    df_input = df_input.astype(float)

    df_input['Long-short Ratio'] = df

    params_5m = {
        'symbol': ticker,
        'period': '5m',
        'limit': 1
    }

    output_5m = client._request('get', targ_url, signed=False, data=params_5m)
    df_5m = pd.DataFrame(output_5m)

    # Converting UNIX stamps to datetimes
    df_5m['Close time'] = df_5m['timestamp']
    df_5m['Close time'] = df_5m['Close time'] / 1000
    df_5m['Close time'] = pd.to_datetime(df_5m['Close time'], unit='s')
    df_5m['Close time'] = df_5m['Close time'].dt.round('1min')

    df_5m = df_5m.set_index('Close time')
    df_5m = df_5m['longShortRatio']
    df_5m = df_5m.astype(float)
    df_input['Long-short Ratio'][-1] = df_5m[-1]

    df = df_input.dropna()

    df.rename(columns = {'Long-short Ratio': 'LSR'}, inplace = True)
    df['LPCT'] = df['LSR'] / (df['LSR'] + 1)

    return df
