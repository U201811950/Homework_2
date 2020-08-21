import baostock as bs
import pandas as pd

####登录系统####
lg = bs.login()
#显示登录返回信息
print('login respond error_code:'+lg.error_code)
print('login respond error_msg:'+lg.error_msg)

#####获取股票历史K线数据####
#详细指标参数
rs = bs.query_history_k_data_plus("sz.002104", "date,open,high,low,close,volume,amount", start_date='2015-01-01', end_date='2020-08-10', frequency="d")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus error_msg:'+rs.error_msg)

####打印结果集####
data_list = []
while (rs.error_code == '0') & rs.next():
    #获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

####结果输出到csv文件####
result.to_csv("C:/Users/大可/Desktop/sz.002104_k_data.csv", index=False)
print(result)

####登出系统####
bs.logout
