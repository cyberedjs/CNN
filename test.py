import pybithumb

result = pybithumb.get_transaction_history("BTC", "KRW", limit=20)
print(result)

#coin_list = pybithumb.get_tickers()
#Date = "2020-12-22"
#for coin in coin_list:
    #ohlcv = pybithumb.get_candlestick(coin, chart_instervals='1m')
    #ohlcv.to_excel("./ohlcv/%s %s ohlcv.xlsx" % (Date, coin))

#ohlcv = pybithumb.get_candlestick("STRAX", chart_instervals='1m')
#print(ohlcv)
