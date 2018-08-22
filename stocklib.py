import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
import bs4 as bs
import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import progress
from progress.bar import Bar
import requests
import time

alpha_key = '4NDO7RSE54GKN195'
alpha_call_delay = 13

data_dir = 'stock_data'
ticker_dir = data_dir + '/' + '_tickers.csv'

ticker_age_min = 1825  # minimum stock data age in days for use in data compilation

def get_tickers(update_ref=False):
    if (update_ref) or not (os.path.exists(ticker_dir)):
        if not os.path.exists(data_dir):
            print('Creating data repository')
            os.makedirs(data_dir)

        if os.path.exists(ticker_dir):     # will overwrite existing ticker reference
            print('Removing outdated ticker reference')
            os.remove(ticker_dir)

        ticker_list = []
        resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class':'wikitable sortable'})
        bar = Bar('Updating ticker reference', max=len(table.findAll('tr')[1:]))
        for row in table.findAll('tr')[1:]:     # ignore first row of table
            ticker = row.findAll('td')[0].text  # get zeroth column
            ticker_list.append(ticker)
            bar.next()
        bar.finish()

        ticker_ref = pd.DataFrame(ticker_list)
        ticker_ref.to_csv(ticker_dir, index=False, header=False)

    print('Getting ticker reference')
    ticker_ref = pd.read_csv(ticker_dir, header=None)
    ticker_list = ticker_ref[0].tolist()

    return ticker_list

def update_data():
    ts = TimeSeries(key=alpha_key, output_format='pandas')

    tickers = get_tickers()

    tickers_miss = []  # check which ticker data is missing and save to list
    for ticker in tickers:
        if not os.path.exists(data_dir + '/{}.csv'.format(ticker)):
            tickers_miss.append(ticker)

    print('Calling {} missing tickers'.format(len(tickers_miss)))
    print('Estimated time: {}min'.format(round(len(tickers_miss) * alpha_call_delay / 60, 2)))

    bar = Bar('Updating Database', max=len(tickers_miss))
    for ticker in tickers_miss:
        bar.next()
        time.sleep(alpha_call_delay)

        df_raw, df_meta = ts.get_daily(symbol=ticker, outputsize='full')
        df_raw.index.names = ['Date']
        df_raw.rename({'1. open':'Open', '2. high':'High', '3. low':'Low', '4. close':'Close', '5. volume':'Volume'}, axis=1, inplace=True)

        # synthetic features
        df_raw['HML'] = (df_raw['High'] - df_raw['Low']) / df_raw['Low'] * 100  # high minus low volatility indicator
        df_raw['PCT'] = (df_raw['Close'] - df_raw['Open']) / df_raw['Open'] * 100   # percentage change indicator

        df_raw.to_csv(data_dir + '/{}.csv'.format(ticker))
    bar.finish()
    print('Done!')

def get_data(ticker):
    ticker_ref = get_tickers()

    if ticker in ticker_ref:   # check ticker against reference
        print('Fetching ticker data')
        data = pd.read_csv(data_dir + '/{}.csv'.format(ticker), index_col=0)
    else:
        raise ValueError('Ticker {} does not exist in reference'.format(ticker))
    
    return data

def compile_data():
    df_comp = pd.DataFrame()
    tickers = get_tickers()

    bar = Bar('Compiling Database', max=len(tickers))
    for ticker in tickers:
        df_ticker = pd.read_csv(data_dir + '/{}.csv'.format(ticker), index_col=0)
        df_ticker.rename({'Close':ticker}, axis=1, inplace=True)

        if (df_ticker[ticker].count() >= ticker_age_min):  # will disregard any tickers below minimum age
            df_comp = pd.concat([df_comp, df_ticker[ticker]], axis=1, sort=True)
        bar.next()
    bar.finish()
    
    df_comp.dropna(axis=0, inplace=True)    # will drop any missing dates

    return df_comp

def visualize_correlation(df_corr):
    matplotlib.style.use('classic')
    fig1 = plt.figure('Ticker Correlation')
    ax1 = fig1.add_subplot(1,1,1)

    df_corr_data = df_corr.values

    heatmap = ax1.pcolor(df_corr_data, cmap='RdYlGn')
    fig1.colorbar(heatmap)

    ax1.set_xticks(np.arange(df_corr_data.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(df_corr_data.shape[0]) + 0.5, minor=False)

    ax1.invert_yaxis()
    ax1.xaxis.tick_top()

    tickers_column = df_corr.columns
    tickers_row = df_corr.index
    ax1.set_xticklabels(tickers_column)
    ax1.set_yticklabels(tickers_row)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    plt.show()

def calculate_correlation(ticker, bounds_upper=0.95, bounds_lower=-0.85):
    df_ticker = get_data(ticker)
    if df_ticker['Close'].count() < ticker_age_min:
        raise ValueError('Ticker {tckr} does not reach age threshold of {min}({age})'.format(tckr=ticker,min=ticker_age_min,age=df_ticker['Close'].count()))
    
    df_comp = compile_data()
    
    print('Calculating correlation of {}:'.format(ticker))
    df_corr = df_comp.corr()

    df_corr_ticker = pd.DataFrame(df_corr[ticker])

    for index, row in df_corr_ticker.iterrows():
        if (bounds_upper < row[0] < 1):     #  (<1) prevents return of identity ticker
            print('{label}(+): {value}'.format(label=index,value=row[0]))
        
        elif (row[0] < bounds_lower):
            print('{label}(-): {value}'.format(label=index,value=row[0]))

if __name__ == '__main__':
    update_data()
    calculate_correlation('TSLA')

