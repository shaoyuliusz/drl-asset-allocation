from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd

STOCKS_LIST =  ["AAPL", "GE", "JPM", "MSFT", "VOD", "NKE", "NVDA", "MMM"]

def data_loader(stock_names: list[str], start_date='2010-01-01', end_date="2017-01-01") -> pd.DataFrame:
    """
    read yahoo finance data
    Args:
        stock_names: stock abbr. names
    Returns: 
        df: pandas dataframe of stock price data
    """
    yf.pdr_override()
    df = pdr.get_data_yahoo(stock_names, start = start_date, end = end_date)
    
    return df

def data_cleaner(df: pd.DataFrame, train_pct:float = 0.8) -> tuple([pd.DataFrame, pd.DataFrame]):
    """clean loaded data"""
    
    df['Adj Close'] = df['Adj Close'].ffill()
    df['Adj Close'] = df['Adj Close'].bfill()
    df = df['Adj Close']
    
    samples_train = int(train_pct*len(df))
    data_train = df[:samples_train]
    data_test = df[samples_train:]

    return (data_train, data_test)

if __name__ == "__main__":
    df_stock = data_loader(STOCKS_LIST)
    df_train, df_val = data_cleaner(df_stock)
    df_test = data_loader(STOCKS_LIST, start_date='2017-01-01', end_date="2018-05-30")

    df_train.to_csv("data/yahoo_finance_train.csv", index=False)
    df_val.to_csv("data/yahoo_finance_val.csv", index=False)
    df_test.to_csv("data/yahoo_finance_test.csv", index=False)

    

