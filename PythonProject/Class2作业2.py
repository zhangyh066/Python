import time

import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'#显示所有运行结果

#设置行不限制数量
#pd.set_option('display.max_rows',None)比较大所以不用

#设置列不限制数量
pd.set_option('display.max_columns', None)
#%%time
data = pd.read_csv('datasets/000001.csv')
data['Day'] = pd.to_datetime(data['Day'],format='%Y/%m/%d')
data.set_index('Day', inplace = True)
data.sort_values(by = ['Day'],axis=0, ascending=True)
print(data)
data_new = data['1995-01':'2025-08'].copy()
data_new['Close'] = pd.to_numeric(data_new['Close'])
data_new['Preclose'] = pd.to_numeric(data_new['Preclose'])
print(data_new)
data_new['Raw_return'] = data_new['Close'] / data_new['Preclose'] - 1
data_new['Log_return'] = np.log(data_new['Close']) - np.log(data_new['Preclose'])
data_new['Pct_change_return'] = data_new['Close'].pct_change()
data_new['Apply_return'] = data_new.apply(lambda row: row['Close'] / row['Preclose'] - 1, axis=1)
data_new['Diff_div_return'] = data_new['Close'].diff() / data_new['Close'].shift(1)
print(data_new)
# 创建新列存储结果
# PS:for/while 语句运算太慢，吃配置所以全部打#号
#if 'Loop_return' not in data_new.columns:
    #data_new['Loop_return'] = np.nan

# 使用for循环计算
#for i in range(len(data_new)):
    #data_new.iloc[i, data_new.columns.get_loc('Loop_return')] = data_new.iloc[i, data_new.columns.get_loc('Close')] / data_new.iloc[i, data_new.columns.get_loc('Preclose')] - 1

# 方法6：使用zip和enumerate组合（比纯for循环更Pythonic）
#close_values = data_new['Close'].values
#preclose_values = data_new['Preclose'].values
#loop_return_values = []

#for i, (close, preclose) in enumerate(zip(close_values, preclose_values)):
    #if preclose != 0 and not np.isnan(preclose):
        #loop_return_values.append(close / preclose - 1)
    #else:
        #loop_return_values.append(np.nan)

#data_new['Loop_return2'] = loop_return_values

# 方法7：使用numpy的向量化操作（高效且简洁）
data_new['Numpy_return'] = (data_new['Close'].values / data_new['Preclose'].values) - 1
print(data_new)
# 方法1：使用resample函数计算月度对数收益率并转换为原始收益率
# 这种方法适合对数收益率，因为对数收益率可以直接相加
Month_data1 = data_new.resample('ME')['Log_return'].sum().to_frame(name='Log_return')
Month_data1['Raw_Return'] = np.exp(Month_data1['Log_return']) - 1
# 添加年月信息便于分析
Month_data1['Year'] = Month_data1.index.year
Month_data1['Month'] = Month_data1.index.month
print(Month_data1.head())
# 方法2：使用resample取月末价格计算月度收益率
# 这种方法直接使用月末价格计算收益率，更符合金融实践
Month_data2 = data_new.resample('ME')['Close'].last().to_frame()
Month_data2['Preclose'] = Month_data2['Close'].shift(1)
Month_data2['Raw_return'] = Month_data2['Close'] / Month_data2['Preclose'] - 1
Month_data2['Log_return'] = np.log(Month_data2['Close']) - np.log(Month_data2['Preclose'])

# 添加年月信息
Month_data2['Year'] = Month_data2.index.year
Month_data2['Month'] = Month_data2.index.month
print(Month_data2.head())
# “1990-12-12”日期格式 里面的year年份 month月份 day 直接提出取来
data_new2 = data_new.copy()
data_new2['year'] = data_new2.index.year
data_new2['month'] = data_new2.index.month
print(data_new2)
# 使用的时间、日期格式提取 字符串提出的方式 前四个字符当作年份 6-7字符是月份 提取出来的是字符串 变成数值
# 方法3：使用groupby函数按年月分组计算月度收益率
# 首先提取年月信息
data_new3 = data_new.copy()
data_new3['year'] = data_new3.index.year
data_new3['month'] = data_new3.index.month

# 使用groupby按年月分组，然后对每组的对数收益率求和
Month_data3 = data_new3.groupby(['year', 'month'])['Log_return'].sum().to_frame()
Month_data3['Raw_Return'] = np.exp(Month_data3['Log_return']) - 1

print(Month_data3)
# 方法4：使用apply和lambda函数进行更灵活的分组计算
# 这种方法可以对每个月的数据进行更复杂的操作
Month_data4 = pd.DataFrame(
    data_new3.groupby(['year', 'month'])['Log_return'].apply(lambda x: sum(x)))
Month_data4.columns = ['Log_return']
Month_data4['Raw_Return'] = np.exp(Month_data4['Log_return']) - 1

# 方法5：使用agg函数同时计算多个统计量
Month_data5 = data_new3.groupby(['year', 'month']).agg({
    'Log_return': ['sum', 'mean', 'std', 'count'],
    'Raw_return': ['mean', 'std']
})

# 显示结果
print("方法4结果:")
print(Month_data4.head())
print("\n方法5结果 (包含多个统计量):")
print(Month_data5.head())
# 计算季度收益率
# 方法1：使用resample函数的'QE'参数（季度末）
Quarter_data1 = data_new.resample('QE')['Log_return'].sum().to_frame(name='Log_return')
Quarter_data1['Raw_Return'] = np.exp(Quarter_data1['Log_return']) - 1
Quarter_data1['Year'] = Quarter_data1.index.year
Quarter_data1['Quarter'] = Quarter_data1.index.quarter

# 方法2：使用季度末价格计算
Quarter_data2 = data_new.resample('QE')['Close'].last().to_frame()
Quarter_data2['Preclose'] = Quarter_data2['Close'].shift(1)
Quarter_data2['Raw_return'] = Quarter_data2['Close'] / Quarter_data2['Preclose'] - 1
Quarter_data2['Log_return'] = np.log(Quarter_data2['Close']) - np.log(Quarter_data2['Preclose'])

# 显示结果
print("季度对数收益率汇总:")
print(Quarter_data1)
print("\n季度末价格计算的收益率:")
print(Quarter_data2)
# 计算年度收益率
# 方法1：使用resample函数的'YE'参数（年末）
Year_data1 = data_new.resample('YE')['Log_return'].sum().to_frame(name='Log_return')
Year_data1['Raw_Return'] = np.exp(Year_data1['Log_return']) - 1

# 方法2：使用年末价格计算
Year_data2 = data_new.resample('YE')['Close'].last().to_frame()
Year_data2['Preclose'] = Year_data2['Close'].shift(1)
Year_data2['Raw_return'] = Year_data2['Close'] / Year_data2['Preclose'] - 1
Year_data2['Log_return'] = np.log(Year_data2['Close']) - np.log(Year_data2['Preclose'])

# 方法3：使用groupby按年分组
data_new4 = data_new.copy()
data_new4['year'] = data_new4.index.year
Year_data3 = data_new4.groupby('year')['Log_return'].sum().to_frame()
Year_data3['Raw_Return'] = np.exp(Year_data3['Log_return']) - 1

# 显示结果
print("年度对数收益率汇总:")
print(Year_data1)
print("\n年末价格计算的收益率:")
print(Year_data2)
print("\n使用groupby计算的年度收益率:")
print(Year_data3)
# 计算滚动收益率（例如：过去30天、60天、90天的收益率 注意这里指的是前30个观测值）

# 方法1：使用rolling窗口函数计算滚动对数收益率之和
rolling_returns = pd.DataFrame()
for window in [5, 10, 20, 30, 60]:
    # 计算滚动窗口的对数收益率之和
    rolling_log_return = data_new['Log_return'].rolling(window=window).sum()
    # 转换为原始收益率
    rolling_returns[f'Rolling_{window}d_Return'] = np.exp(rolling_log_return) - 1

# 方法2：使用pct_change计算滚动价格变化
rolling_price_returns = pd.DataFrame()
for window in [5, 10, 20, 30, 60]:
    rolling_price_returns[f'Rolling_{window}d_Price_Return'] = data_new['Close'].pct_change(periods=window)

print("滚动收益率 (基于对数收益率累加):")
print(rolling_returns.tail())
print("\n滚动收益率 (基于价格变化):")
print(rolling_price_returns.tail())
# 计算累积收益率
# 累积收益率用于观察长期投资表现，从某个起始点开始累积

# 方法1：使用对数收益率累加后转换
# 这是最准确的方法，特别是对于长期累积
cumulative_returns = pd.DataFrame()
cumulative_returns['Cumulative_Log_Return'] = data_new['Log_return'].cumsum()
cumulative_returns['Cumulative_Return'] = np.exp(cumulative_returns['Cumulative_Log_Return']) - 1

# 方法2：使用cumprod函数直接累乘(1+r)
# 这种方法在金融实践中也很常见
cumulative_returns['Cumulative_Return_Prod'] = (1 + data_new['Raw_return']).cumprod() - 1

# 方法3：使用pandas的累积函数
cumulative_returns['Cumulative_Return_Alt'] = data_new['Raw_return'].add(1).cumprod().sub(1)

print("不同方法计算的累积收益率:")
print(cumulative_returns)