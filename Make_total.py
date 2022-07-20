import numpy as np
import pandas as pd
import pybithumb
import os
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from Funcs_CNN4 import cmo, rsi, obv, macd
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

home_dir = os.path.expanduser('~Admin/')
dir = home_dir + 'Desktop/tryout/CNN/ohlcv/'
ohlcv_list = os.listdir(dir)

def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def low_high(Coin, input_data_length, ip_limit=None, trade_limit=None, crop_size=None, lowhigh_point=None):

    #   거래 제한은 고점과 저점을 분리한다.

    #   User-Agent Configuration
    #   IP - Change
    if ip_limit is None:
        ohlcv_excel = pybithumb.get_candlestick(Coin, chart_intervals='1m')
    else:
        ohlcv_excel = pybithumb.get_candlestick(Coin, chart_intervals='1m')

    # price_gap = ohlcv_excel.close.max() / ohlcv_excel.close.min()
    # if (price_gap < 1.07) and (trade_limit is not None):
    #     return None, None, None, None

    ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel)
    macd(ohlcv_excel, short=105, long=168, signal=32)
    print(len(ohlcv_excel))
    # quit()

    check_span = 30
    trade_state = [np.NaN] * len(ohlcv_excel)
    for i in range(len(ohlcv_excel) - check_span):
        #   저점
        if ohlcv_excel['close'][i + 1:i + 1 + check_span].min() >= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[0] <= 3:
                trade_state[i] = 1
            else:
                trade_state[i] = 0
        #   고점
        elif ohlcv_excel['close'][i + 1:i + 1 + check_span].max() <= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[-1] <= 3:
                trade_state[i] = 2
            else:
                trade_state[i] = 0
        #   거래 안함
        else:
            trade_state[i] = 0

    ohlcv_excel['trade_state'] = trade_state

    closeprice = ohlcv_excel['close'].iloc[-1]
    datetime_list = ohlcv_excel.index.values

    # ----------- dataX, dataY 추출하기 -----------#
    #   OBV :
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   price ma
        ohlcv_data[:, [0, 1, 2, 3]] = max_abs_scaler(ohlcv_data[:, [0, 1, 2, 3]])
        # volume
        ohlcv_data[:, [4]] = max_abs_scaler(ohlcv_data[:, [4]])
        #   CMO
        ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
        #   OBV
        ohlcv_data[:, [-7]] = max_abs_scaler(ohlcv_data[:, [-7]])
        #   RSI
        ohlcv_data[:, [-6]] = max_abs_scaler(ohlcv_data[:, [-6]])
        #   MACD
        ohlcv_data[:, -5:-1] = max_abs_scaler(ohlcv_data[:, -5:-1])

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_x = ohlcv_data[i - crop_size: i]
            price = group_x[:, :4]
            volume = group_x[:, [4]]

            CMO = group_x[:, [-8]]
            OBV = group_x[:, [-7]]
            RSI = group_x[:, [-6]]
            MACD = group_x[:, -5:-1]

            x = np.concatenate(
                (price, volume, CMO, OBV, RSI, MACD), axis=1)
            group_x = x[-input_data_length:]

            dataX.append(group_x)  # dataX 리스트에 추가

        return dataX, closeprice, ohlcv_data[crop_size:, ], datetime_list

def made_x(file, input_data_length, model_num, check_span, get_fig, crop_size=None, sudden_death=0):

    ohlcv_excel = pd.read_excel(dir + file, index_col=0)

    ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel)
    macd(ohlcv_excel, short=105, long=168, signal=32)

    # print(ohlcv_excel)
    # quit()

    #   이후 check_span 데이터와 현재 포인트를 비교해서 현재 포인트가 저가인지 고가인지 예측한다.
    #   진입, 저점, 고점, 거래 안함의 y_label 인 trade_state  >> [1, 2, 0]
    #   저점과 고점은 최대 3개의 중복 값을 허용한다.
    trade_state = [np.nan] * len(ohlcv_excel)
    for i in range(len(ohlcv_excel) - check_span):
        #   저점
        if ohlcv_excel['close'][i + 1:i + 1 + check_span].min() >= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[0] <= 3:
                trade_state[i] = 1
            else:
                trade_state[i] = 0
        #   고점
        elif ohlcv_excel['close'][i + 1:i + 1 + check_span].max() <= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[-1] <= 3:
                trade_state[i] = 2
            else:
                trade_state[i] = 0
        #   거래 안함
        else:
            trade_state[i] = 0

    ohlcv_excel['trade_state'] = trade_state
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()): -check_span].astype(np.float)

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #

        #    X_data    #
        # price ma
        ohlcv_data[:, [0, 1, 2, 3]] = max_abs_scaler(ohlcv_data[:, [0, 1, 2, 3]])
        # volume
        ohlcv_data[:, [4]] = max_abs_scaler(ohlcv_data[:, [4]])
        #   CMO
        ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
        #   OBV
        ohlcv_data[:, [-7]] = max_abs_scaler(ohlcv_data[:, [-7]])
        #   RSI
        ohlcv_data[:, [-6]] = max_abs_scaler(ohlcv_data[:, [-6]])
        #   MACD
        ohlcv_data[:, -5:-1] = max_abs_scaler(ohlcv_data[:, -5:-1])

        #    Y_data    #
        trade_state = ohlcv_data[:, [-1]]
        y = trade_state

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_x = ohlcv_data[i - crop_size: i]
            group_y = y[i]
            price = group_x[:, :4]
            volume = group_x[:, [4]]
            CMO = group_x[:, [-8]]
            OBV = group_x[:, [-7]]
            RSI = group_x[:, [-6]]
            MACD = group_x[:, -5:-1]

            x = np.concatenate((price, volume, CMO, OBV, RSI, MACD), axis=1)

            group_x = x[-input_data_length:]
            # print(group_x[0])
            # quit()

            #   데이터 값에 결측치가 존재하는 경우 #
            if sum(sum(np.isnan(group_x))) > 0:
                continue

            dataX.append(group_x)  # dataX 리스트에 추가
            dataY.append(group_y)

        #                      Get Figure                     #
        if get_fig == 1:
            spanlist_low = []
            spanlist_high = []

            for m in range(len(trade_state)):
                if (trade_state[m] > 0.5) and (trade_state[m] < 1.5):
                    if m + 1 < len(trade_state):
                        spanlist_low.append((m, m + 1))
                    else:
                        spanlist_low.append((m - 1, m))

            for m in range(len(trade_state)):
                if (trade_state[m] > 1.5) and (trade_state[m] < 2.5):
                    if m + 1 < len(trade_state):
                        spanlist_high.append((m, m + 1))
                    else:
                        spanlist_high.append((m - 1, m))

            # ----------- 인덱스 초기화 됨 -----------#

            # ----------- 공통된 Chart 그리기 -----------#

            plt.subplot(211)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            # plt.plot(scaled_OBV, 'b', label='OBV')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_low)):
                plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.5)

            plt.subplot(212)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            # plt.plot(scaled_OBV, 'b', label='OBV')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_high)):
                plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.5)

            Date = file.split()[0]
            Coin = file.split()[1].split('.')[0]
            # plt.savefig('./Figure_data/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
            # plt.close()
            plt.show()
            # ----------- Chart 그리기 -----------#

        return dataX, dataY

if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 54
    model_num = 117
    check_span = 30
    get_fig = 0

    #       Make folder      #
    try:
        os.mkdir('./Figure_data/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass

    Made_X = []
    Made_Y = []

    ohlcv_list = ['2020-12-22 QKC ohlcv.xlsx']

    for file in ohlcv_list:

        result = made_x(file, input_data_length, model_num, check_span, get_fig, crop_size=input_data_length)

        # result = low_high('dvp'.upper(), input_data_length, lowhigh_point='on')
        # print(result)
        # quit()

        # ------------ 데이터가 있으면 dataX, dataY 병합하기 ------------#
        if result is not None:
            if result[0] is not None:

                Made_X += result[0]
                Made_Y += result[1]

            # 누적 데이터량 표시
            print(file, len(Made_X))

    # SAVING X, Y
    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), np.array(Made_X))
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), np.array(Made_Y))
