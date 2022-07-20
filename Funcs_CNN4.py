import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

def cmo(df, period=9):

    df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
    df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
    # print(df)

    df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
            df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

    del df['closegap_cunsum']
    del df['closegap_abs_cumsum']

    return df['CMO']


def rsi(ohlcv_df, period=14):

    ohlcv_df['up'] = np.where(ohlcv_df.diff(1)['close'] > 0, ohlcv_df.diff(1)['close'], 0)
    ohlcv_df['down'] = np.where(ohlcv_df.diff(1)['close'] < 0, ohlcv_df.diff(1)['close'] * (-1), 0)
    ohlcv_df['au'] = ohlcv_df['up'].rolling(period).mean()
    ohlcv_df['ad'] = ohlcv_df['down'].rolling(period).mean()
    ohlcv_df['RSI'] = ohlcv_df.au / (ohlcv_df.ad + ohlcv_df.au) * 100

    del ohlcv_df['up']
    del ohlcv_df['down']
    del ohlcv_df['au']
    del ohlcv_df['ad']

    return ohlcv_df.RSI


def obv(ohlcv_excel):

    obv = [0] * len(ohlcv_excel)
    for m in range(1, len(ohlcv_excel)):
        if ohlcv_excel['close'].iloc[m] > ohlcv_excel['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + ohlcv_excel['volume'].iloc[m]
        elif ohlcv_excel['close'].iloc[m] == ohlcv_excel['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - ohlcv_excel['volume'].iloc[m]

    return obv


def macd(ohlcv_excel, short=105, long=168, signal=32):

    ohlcv_excel['MACD'] = ohlcv_excel['close'].ewm(span=short, min_periods=short-1, adjust=False).mean() - \
        ohlcv_excel['close'].ewm(span=long, min_periods=long-1, adjust=False).mean()
    ohlcv_excel['MACD_Signal'] = ohlcv_excel['MACD'].ewm(span=signal, min_periods=signal-1, adjust=False).mean()
    ohlcv_excel['MACD_OSC'] = ohlcv_excel.MACD - ohlcv_excel.MACD_Signal
    ohlcv_excel['MACD_Zero'] = 0

    return


def ema_ribbon(ohlcv_excel, ema_1=5, ema_2=8, ema_3=13):

    ohlcv_excel['EMA_1'] = ohlcv_excel['close'].ewm(span=ema_1, min_periods=ema_1-1, adjust=False).mean()
    ohlcv_excel['EMA_2'] = ohlcv_excel['close'].ewm(span=ema_2, min_periods=ema_2-1, adjust=False).mean()
    ohlcv_excel['EMA_3'] = ohlcv_excel['close'].ewm(span=ema_3, min_periods=ema_3-1, adjust=False).mean()

    return


def ema_cross(ohlcv_excel, ema_1=30, ema_2=60):

    ohlcv_excel['EMA_1'] = ohlcv_excel['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    ohlcv_excel['EMA_2'] = ohlcv_excel['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()

    return


def profitage(Coin, input_data_length, model_num, wait_tick=3, over_tick=10, Date='2019-09-25', excel=0):

    try :
        df = pd.read_excel('./pred_ohlcv/%s_%s/%s %s ohlcv.xlsx' % (input_data_length, model_num, Date, Coin), index_col=0)

    except Exception as e:
        print('Error in loading ohlcv_data :', e)
        return 1.0, 1.0, 1.0

    # 매수 시점 = 급등 예상시, 매수가 = 이전 종가
    df['BuyPrice'] = np.where((df['trade_state'] > 0.5) & (df['trade_state'] < 1.5), df['close'].shift(1), np.nan)
    # df['BuyPrice'] = df['BuyPrice'].apply(clearance)

    # 거래 수수료
    fee = 0.005

    # DatetimeIndex 를 지정해주기 번거로운 상황이기 때문에 틱을 기준으로 거래한다.
    # DatetimeIndex = df.axes[0]

    # ------------------- 상향 / 하향 매도 여부와 이익률 계산 -------------------#

    # high 가 SPP 를 건드리거나, low 가 SPM 을 건드리면 매도 체결 [ 매도 체결될 때까지 SPP 와 SPM 은 유지 !! ]
    length = len(df.index) - 1  # 데이터 갯수 = 1, m = 0  >> 데이터 갯수가 100 개면 m 번호는 99 까지 ( 1 - 100 )

    if length < 100:
        return 1.0, 1.0, 1.0

    # 병합할 dataframe 초기화
    bprelay = pd.DataFrame(index=np.arange(len(df)), columns=['bprelay'])
    condition = pd.DataFrame(index=np.arange(len(df)), columns=["Condition"])
    Profits = pd.DataFrame(index=np.arange(len(df)), columns=["Profits"])
    price_point = pd.DataFrame(index=np.arange(len(df)), columns=['Price_point'])

    Profits.Profits = 1.0
    Minus_Profits = 1.0

    # 오더라인과 매수가가 정해진 곳에서부터 일정시간까지 오더라인과 매수가를 만족할 때까지 대기  >> 일정시간,
    m = 0
    while m <= length:

        while True:  # bp 찾기
            if pd.notnull(df.iloc[m]['BuyPrice']):
                break
            m += 1
            if m > length:
                break

        if (m > length) or pd.isnull(df.iloc[m]['BuyPrice']):
            break
        bp = df.iloc[m]['BuyPrice']

        #       매수 등록 완료, 매수 체결 대기     #

        start_m = m
        while True:
            bprelay["bprelay"].iloc[m] = bp

            # 매수 체결 조건
            if df.iloc[m]['low'] <= bp and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
                # print(df['low'].iloc[m], '<=', bp, '?')

                condition.iloc[m] = '매수 체결'
                bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

                m += 1
                break
            else:
                m += 1
                if m > length:
                    break
                if m - start_m >= wait_tick:
                    break

                condition.iloc[m] = '매수 대기'
                bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

    # df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)

    # if excel == 1:
    #     df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
    #     quit()

    #           지정 매도가 표시 완료           #

    #                               수익성 검사                                  #

    m = 0
    spanlist = []
    spanlist_limit = []
    spanlist_breakaway = []
    while m <= length:  # 초반 시작포인트 찾기

        while True: # SPP 와 SPM 찾긴
            if condition.iloc[m]['Condition'] == '매수 체결':
                # and type(df.iloc[m]['SPP']) != str:  # null 이 아니라는 건 오더라인과 매수가로 캡쳐했다는 거
                break
            m += 1
            if m > length: # 차트가 끊나는 경우, 만족하는 spp, spm 이 없는 경우
                break

        if (m > length) or pd.isnull(condition.iloc[m]['Condition']):
            break

        start_m = m
        start_span = df.index[m]

        #           매수 체결 지점 확인          #

        #       Predict value == 1 지점으로 부터 일정 시간 (10분) 지난 후 checking      #
        while m - start_m <= over_tick:
            m += 1
            if m > length:
                break
            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

        if m > length:
            break

        while True:

            #       고점 예측시 매도       #
            if df.iloc[m]['trade_state'] > 1.5:
                break

            m += 1
            if m > length:
                break

            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

        if m > length:
            # print(condition.iloc[m - 1])
            if condition.iloc[m - 1].item() == '매도 대기':
                print('매수 체결 / 매도 신호 미포착 : ', Date, Coin)
                try:
                    spanlist.append((start_m, m - 1))
                    spanlist_breakaway.append((start_span, df.index[m - 1]))

                except Exception as e:
                    pass
            break

        elif df.iloc[m]['trade_state'] > 1.5:
            condition.iloc[m] = "매도 체결"
            Profits.iloc[m] = df.iloc[m]['close'] / bprelay["bprelay"].iloc[m - 1] - fee

            if float(Profits.iloc[m]) < 1:
                Minus_Profits *= float(Profits.iloc[m])
                try:
                    spanlist.append((start_m, m))
                    spanlist_breakaway.append((start_span, df.index[m]))

                except Exception as e:
                    pass

            else:
                try:
                    spanlist.append((start_m, m))
                    spanlist_limit.append((start_span, df.index[m]))

                except Exception as e:
                    pass

                if float(Profits.iloc[m]) > 1.05:
                    price_point.iloc[m] = df['high'][:start_m].max() / df['low'][:start_m].min()

        # 체결시 재시작
        m += 1

    df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, price_point, how='outer', left_index=True, right_index=True)

    if excel == 1:
        df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))

    profits = Profits.cumprod()  # 해당 열까지의 누적 곱!

    if np.isnan(profits.iloc[-1].item()):
        return 1.0, 1.0, 1.0

    # [-1] 을 사용하려면 데이터가 존재해야 되는데 데이터 전체가 결측치인 경우가 존재한다.
    if len(profits) == 0:
        return 1.0, 1.0, 1.0

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits


if __name__=="__main__":
    # Best Value #########################################################
    # ------- FEED MACHINE -------#
    excel_file = input("Input File : ")

    # ----------- PARAMETER -----------#
    Coin = excel_file.split()[1].split('.')[0]
    Date = excel_file.split()[0]
    input_data_length = 54
    # model_num = input('Press model num : ')
    model_num = 21
    wait_tick = 3
    over_tick = 10
    ######################################################################

    print(profitage(Coin, input_data_length, model_num, wait_tick, over_tick, Date, 1))
