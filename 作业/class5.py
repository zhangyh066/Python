import numpy as np # 数据处理最重要的模块
import pandas as pd # 数据处理最重要的模块
import scipy.stats as stats # 统计模块
import scipy
# import pymysql  # 导入数据库模块

from datetime import datetime # 时间模块
import statsmodels.formula.api as smf  # OLS regression

# import pyreadr # read RDS file

from matplotlib import style
import matplotlib.pyplot as plt  # 画图模块
import matplotlib.dates as mdates


from matplotlib.font_manager import FontProperties # 作图中文
from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['font.family'] = 'Times New Roman'


import matplotlib.pyplot as plt
import io
from IPython.display import SVG, display

# 生成 SVG 直接嵌到 PyCharm SciView 或控制台
plt.plot([0, 1], [0, 1])
buf = io.BytesIO()
plt.savefig(buf, format='svg', bbox_inches='tight')
buf.seek(0)
display(SVG(buf.read()))


from IPython.core.interactiveshell import InteractiveShell # jupyter运行输出的模块
#显示每一个运行结果
InteractiveShell.ast_node_interactivity = 'all'

#设置行不限制数量
#pd.set_option('display.max_rows',None)

#设置列不限制数量
pd.set_option('display.max_columns', None)

from pandas.tseries.offsets import MonthEnd # 月末
Market_ret = pd.read_csv('datasets/Marketret_mon_stock2024.csv')
Market_ret['month'] = pd.to_datetime(Market_ret['month'], format='%b %Y') + MonthEnd(0)
Market_ret.set_index('month', inplace=True)
Market_ret.sort_index(inplace=True)
Market_ret = Market_ret.drop(columns=['Unnamed: 0'])
print(Market_ret)

inflation = pd.read_csv('datasets/inflation.csv')
inflation['month'] = pd.to_datetime(inflation['month'],format='%Y/%m/%d')
inflation.set_index('month',inplace=True)
print(inflation)

price_dividend = pd.read_csv('datasets/Price_dividend_mon2024.csv')
price_dividend['month'] = pd.date_range(start='1990-12-31', end='2024-12-31', freq='ME')
price_dividend.set_index('month', inplace=True)
price_dividend.sort_index(inplace=True)
price_dividend = price_dividend.drop(columns=['Unnamed: 0'])
print(price_dividend)

price_earning = pd.read_csv('datasets/Price_earnings_mon2024.csv')
price_earning['month'] = pd.date_range(start='1991-01-31', end='2024-12-31', freq='ME')
price_earning.set_index('month', inplace=True)
price_earning.sort_index(inplace=True)
print(price_earning)

price_bookvalue = pd.read_csv('datasets/Price_bookvalue_mon2024.csv')
price_bookvalue['month'] = pd.date_range(start='1990-12-31', end='2024-12-31', freq='ME')
price_bookvalue.set_index('month', inplace=True)
price_bookvalue.sort_index(inplace=True)
print(price_bookvalue)

reg_data = pd.merge(Market_ret,price_dividend,on = 'month')
reg_data = pd.merge(reg_data,price_earning,on = 'month')
reg_data = pd.merge(reg_data,price_bookvalue,on='month')
reg_data = pd.merge(reg_data,inflation,on='month')
reg_data = reg_data[reg_data.index >= '1995-01-31']
print(reg_data)

reg_data['pe'].describe().round(5)

reg_data['pb'].skew()
reg_data['pb'].kurtosis()

fig = plt.figure(figsize=(10, 5)) # 图片比例
ax = fig.add_subplot(1, 1, 1)  # 定义ax
ax.plot(
    'pe',  # 要画图的变量名
    '.-g',  # 线的类型
    linewidth = 1,  # 线的粗细
    data = reg_data['1995-01-01':'2024-12-31'])  # 画图的数据
plt.title("China's Stock Market Price Earnings Ratio") # 画图的标题
plt.xlabel('Month') # 画图的x轴名称
plt.ylabel('PE') # 画图的y轴名称

# 设置x轴的日期显示格式
data_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(data_format)
ax.xaxis.set_major_locator(mdates.YearLocator())

# 转置x轴的日期显示格式
plt.xticks(rotation = 90)

plt.show();

fig = plt.figure(figsize=(10, 5)) # 图片比例
ax = fig.add_subplot(1, 1, 1)  # 定义ax
ax.plot(
    'pe',  # 要画图的变量名
    '.-b',  # 线的类型
    linewidth = 1,  # 线的粗细
    data = reg_data['1995-01-01':'2023-12-31'])  # 画图的数据
ax.plot(
    'pb',  # 要画图的变量名
    '.-y',  # 线的类型
    linewidth = 1,  # 线的粗细
    data = reg_data['1995-01-01':'2023-12-31'])  # 画图的数据
ax.plot(
    'pd',  # 要画图的变量名
    '.-g',  # 线的类型
    linewidth = 1,  # 线的粗细
    data = reg_data['1995-01-01':'2023-12-31'])  # 画图的数据

plt.title("China's Stock Market Price Ratio") # 画图的标题
plt.xlabel('Month') # 画图的x轴名称


# 设置x轴的日期显示格式
data_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(data_format)
ax.xaxis.set_major_locator(mdates.YearLocator())

# 转置x轴的日期显示格式
plt.xticks(rotation = 90)

plt.legend()

plt.show();

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 1, 1)  #(x, x, x)这里前两个表示几*几的网格，最后一个表示第几子图

ax1.plot(reg_data['pd'],
         color='pink',
         marker='.',
         linestyle='-',
         linewidth=1.2,
         markersize=6,
         alpha=0.4,
         label='pd')
ax1.set_xlabel('month')  # 设置横坐标标签
ax1.set_ylabel('Price Ratio')  # 设置左边纵坐标标签
ax1.set_title("Price Ratio and Return: Monthly 2000-2021")  # 给整张图命名

ax2 = ax1.twinx()  #twinx()函数表示共享x轴
ax2.plot(reg_data['ret'],
         color='green',
         marker='o',
         linestyle='-',
         linewidth=1.2,
         markersize=2,
         alpha=0.7,
         label='Return')
ax2.set_ylabel('Return')  # 设置右边纵坐标标签

# 设置x轴的日期显示格式
data_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(data_format)
ax.xaxis.set_major_locator(mdates.YearLocator())

# 转置x轴的日期显示格式
plt.xticks(rotation = 90)

lines,labels = ax1.get_legend_handles_labels()
lines2,labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines+lines2,labels+labels2,loc='upper right')

plt.show();

from statsmodels.tsa.stattools import adfuller as ADF

# 对月收益率数据进行ADF检验
adf_result = ADF(reg_data[reg_data.index >= '2000-01-31']['pd'])

print('原始序列的ADF检验结果:')
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'   {key}: {value:.4f}')

if adf_result[1] <= 0.05:
    print('结论: p-value小于0.05，拒绝原假设，序列是平稳的。')
else:
    print('结论: p-value大于0.05，未能拒绝原假设，序列是非平稳的。')

reg_data['lpd'] = reg_data['pd'].shift(1)
model_fore_pd = smf.ols('ret ~ lpd',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_fore_pd.summary())

reg_data['lcpi'] = reg_data['cpi'].shift(2)
model_cpipd = smf.ols('ret ~ lcpi + lpd',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_cpipd.summary())

print(reg_data)

reg_data['fitted_ret'] = model_cpipd.fittedvalues
print(reg_data)

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)  #(x, x, x)这里前两个表示几*几的网格，最后一个表示第几子图

ax.plot(reg_data['ret'],
         color='green',
         marker='.',
         linestyle='-',
         linewidth=1,
         markersize=6,
         alpha=0.4,
         label='Market Return')
plt.xlabel('month')  # 设置横坐标标签
plt.ylabel('Return')  # 设置左边纵坐标标签
#plt.legend(loc=2)  # 设置图例在左上方
plt.title("Return: Monthly 1995-2021")  # 给整张图命名

# ax2 = ax1.twinx()  #twinx()函数表示共享x轴
ax.plot(reg_data['fitted_ret'],
         color='blue',
         marker='o',
         linestyle='-',
         linewidth=1,
         markersize=2,
         alpha=0.7,
         label='fitted_ret')
# ax2.set_ylabel('fitted_mv')  # 设置右边纵坐标标签
# ax2.legend(loc=1)  # 设置图例在右上方

# 设置x轴的日期显示格式
data_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(data_format)
ax.xaxis.set_major_locator(mdates.YearLocator())

# 转置x轴的日期显示格式
plt.xticks(rotation = 90)

plt.legend()

plt.show();

from statsmodels.iolib.summary2 import summary_col
reg_data['lcpi'] = reg_data['cpi'].shift(2)
reg_data['lpd'] = reg_data['pd'].shift(1)
reg_data['lpe'] = reg_data['pe'].shift(1)
reg_data['lpb'] = reg_data['pb'].shift(1)
info_dict = {'No. observations': lambda x: f"{int(x.nobs):d}"}

model_pd = smf.ols('ret ~ lpd',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
model_pe = smf.ols('ret ~ lpe',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
model_pb = smf.ols('ret ~ lpb',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
model_cpi = smf.ols('ret ~ lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
model_pdto = smf.ols('ret ~ lpd + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
model_all = smf.ols('ret ~ lpd + lpe + lpb + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})

results_table = summary_col(results=[model_pd,model_pe, model_pb, model_cpi, model_pdto,model_all],
                            float_format='%0.3f', #数据显示的格式，默认四位小数
                            stars=True, # 是否有*，True为有
                            model_names=['1', '2', '3','4','5','6'],
                            info_dict=info_dict,
                            regressor_order=['Intercept', 'lpd','lpe','lpb','lcpi'])

results_table.add_title(
    'Table - OLS Regressions: Forecast Monthly Stock Market Return')

print(results_table)

Qreg_data = reg_data.resample('QE').apply({
    'ret':
    lambda x: np.exp(sum(np.log( 1 + x))) - 1,
    'pe':
    lambda x: sum(x),
    'pb':
    lambda x: sum(x),
    'pd':
    lambda x: sum(x),
    'cpi':
    lambda x: sum(x),
})
print(Qreg_data)

Qreg_data['lcpi'] = Qreg_data['cpi'].shift(1)
Qreg_data['lpd'] = Qreg_data['pd'].shift(1)
model_to = smf.ols('ret ~ lpd + lcpi',
                 data=Qreg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
print(model_to.summary())

from statsmodels.iolib.summary2 import summary_col
Qreg_data['lcpi'] = Qreg_data['cpi'].shift(1)
Qreg_data['lpd'] = Qreg_data['pd'].shift(1)
Qreg_data['lpe'] = Qreg_data['pe'].shift(1)
Qreg_data['lpb'] = Qreg_data['pb'].shift(1)

info_dict = {'No. observations': lambda x: f"{int(x.nobs):d}"}

model_pd = smf.ols('ret ~ lpd',
                 data=Qreg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_pe = smf.ols('ret ~ lpe',
                    data=Qreg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_pb = smf.ols('ret ~ lpb',
                    data=Qreg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpi = smf.ols('ret ~ lcpi',
                 data=Qreg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipd = smf.ols('ret ~ lpd + lcpi',
                 data=Qreg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipe = smf.ols('ret ~ lpe + lcpi',
                 data=Qreg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipb = smf.ols('ret ~ lpb + lcpi',
                 data=Qreg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_all = smf.ols('ret ~ lpd + lpe + lpb + lcpi',
                 data=Qreg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})

results_table = summary_col(results=[model_pd,model_pe, model_pb, model_cpi,model_cpipd,model_cpipe,model_cpipb,model_all],
                            float_format='%0.3f', #数据显示的格式，默认四位小数
                            stars=True, # 是否有*，True为有
                            model_names=['1', '2', '3','4','5','6','7','8'],
                            info_dict=info_dict,
                            regressor_order=['Intercept','lpd','lpe','lpb','lcpi'])

results_table.add_title(
    'Table - OLS Regressions: Forecast Quarterly Stock Market Return')

print(results_table)

reg_data.columns

# long horizontal regression table
from statsmodels.iolib.summary2 import summary_col

info_dict = {'No. observations': lambda x: f"{int(x.nobs):d}"}


model_pd = smf.ols('marketret3 ~ lpd',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_pe = smf.ols('marketret3 ~ lpe',
                    data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_pb = smf.ols('marketret3 ~ lpb',
                    data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpi = smf.ols('marketret3 ~ lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipd = smf.ols('marketret3 ~ lpd + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipe = smf.ols('marketret3 ~ lpe + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipb = smf.ols('marketret3 ~ lpb + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_all = smf.ols('marketret3 ~ lpd + lpe + lpb + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})

results_table = summary_col(results=[model_pd,model_pe, model_pb, model_cpi,model_cpipd,model_cpipe,model_cpipb,model_all],
                            float_format='%0.3f', #数据显示的格式，默认四位小数
                            stars=True, # 是否有*，True为有
                            model_names=['1', '2', '3','4','5','6','7','8'],
                            info_dict=info_dict,
                            regressor_order=['Intercept','lpd','lpe','lpb','lcpi'])

results_table.add_title(
    'Table - OLS Regressions: Forecast Long Horizon Stock Market Return')
print(results_table)

# long horizontal regression table
from statsmodels.iolib.summary2 import summary_col
info_dict = {'No. observations': lambda x: f"{int(x.nobs):d}"}


model_pd = smf.ols('marketret6 ~ lpd',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_pe = smf.ols('marketret6 ~ lpe',
                    data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_pb = smf.ols('marketret6 ~ lpb',
                    data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpi = smf.ols('marketret6 ~ lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipd = smf.ols('marketret6 ~ lpd + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipe = smf.ols('marketret6 ~ lpe + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipb = smf.ols('marketret6 ~ lpb + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_all = smf.ols('marketret6 ~ lpd + lpe + lpb + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})

results_table = summary_col(results=[model_pd,model_pe, model_pb, model_cpi,model_cpipd,model_cpipe,model_cpipb,model_all],
                            float_format='%0.3f', #数据显示的格式，默认四位小数
                            stars=True, # 是否有*，True为有
                            model_names=['1', '2', '3','4','5','6','7','8'],
                            info_dict=info_dict,
                            regressor_order=['Intercept','lpd','lpe','lpb','lcpi'])

results_table.add_title(
    'Table - OLS Regressions: Forecast Long Horizon Stock Market Return')
print(results_table)

# long horizontal regression table
from statsmodels.iolib.summary2 import summary_col

info_dict = {'No. observations': lambda x: f"{int(x.nobs):d}"}


model_pd = smf.ols('marketret12 ~ lpd',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_pe = smf.ols('marketret12 ~ lpe',
                    data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_pb = smf.ols('marketret12 ~ lpb',
                    data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpi = smf.ols('marketret12 ~ lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipd = smf.ols('marketret12 ~ lpd + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipe = smf.ols('marketret12 ~ lpe + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_cpipb = smf.ols('marketret12 ~ lpb + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 2})
model_all = smf.ols('marketret12 ~ lpd + lpe + lpb + lcpi',
                 data=reg_data['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})

results_table = summary_col(results=[model_pd,model_pe, model_pb, model_cpi,model_cpipd,model_cpipe,model_cpipb,model_all],
                            float_format='%0.3f', #数据显示的格式，默认四位小数
                            stars=True, # 是否有*，True为有
                            model_names=['1', '2', '3','4','5','6','7','8'],
                            info_dict=info_dict,
                            regressor_order=['Intercept','lpd','lpe','lpb','lcpi'])

results_table.add_title(
    'Table - OLS Regressions: Forecast Long Horizon Stock Market Return')
print(results_table)