# 导入必要的库
import pandas as pd
from util.StochRSI import StochRSI  # 导入StochRSI封装函数
from util.KDJ import KDJ   # 导入KDJ封装函数
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np

# 计算未来收益函数
def fun3_1():
    """
    计算未来收益
    :return:
    """
    df = pd.read_csv('daydata.csv',index_col=0)
    df['r_1'] = (df['close']-df.shift(1)['close'])/df.shift(1)['close']
    df['r_1'] = df['r_1'].shift(-1)

    df['r_5'] = (df['close'] - df.shift(5)['close']) / df.shift(5)['close']
    df['r_5'] = df['r_5'].shift(-5)

    df['r_10'] = (df['close'] - df.shift(10)['close']) / df.shift(10)['close']
    df['r_10'] = df['r_10'].shift(-10)

    df.dropna()
    df.to_csv('raw_factor_data.csv') #将数据导入到新的csv文件里
# fun3_1()

def fun3_2():
    """
    打图看一下收益
    :return:
    """
    df = pd.read_csv('raw_factor_data.csv',index_col=0)
    y = (df['close'] - df.shift(10)['close'])
    plt.plot(y)
    plt.show()
    # 去掉异常数据
    df = df[100:]
    df = df.reset_index(drop=True)
    df.to_csv('factor_data.csv')

# fun3_2()

def fun3_3():
    # 计算因子值
    df = pd.read_csv('factor_data.csv', index_col=0)
    df.rename(columns={"high": "H", "low": "L", 'open': 'O', 'close': 'C'}, inplace=True)
    data = df.to_dict(orient="records")
    kdj = KDJ(12,6,3)
    rsi = StochRSI(9,3,3,3)
    for i, kline in enumerate(data):
        kdj.cal_index(kline)
        rsi.cal_index(kline)
        df.loc[i, 'k'] = kdj.K
        df.loc[i, 'rsi'] = rsi.rsi
    df = df[11:-10]
    df = df.reset_index(drop=True)
    df.to_csv('factor_data.csv')

# fun3_3()

def fun3_4():
    # 看因子是否平稳  不平稳就要做处理
    df = pd.read_csv('factor_data.csv', index_col=0)
    plt.plot(df['rsi'])
    plt.show()

fun3_4()