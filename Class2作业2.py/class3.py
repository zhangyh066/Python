import matplotlib
import pandas as pd
import numpy as np

import scipy.stats as stats
import scipy

from datetime import datetime
import calendar #日历模块
import statsmodels.api as sm
import statsmodels.formula.api as smf

#import pyreadr # R data file reader

from matplotlib import pyplot as plt
from matplotlib import dates as mdates #时序图
from matplotlib import ticker as mticker #y轴刻度 百分比
from matplotlib import style
import seaborn as sns
from matplotlib_inline.config import InlineBackend


from matplotlib.font_manager import FontProperties # 作图中文
from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac系统中文字体
plt.rcParams['font.family'] = 'Arial Unicode MS'
#输出矢量图 渲染矢量图
import matplotlib.pyplot as plt

#是IPython的命令，pycharm中不认识%的语法，因此换一种方式进行
# 等价于 %matplotlib inline
# 在脚本里只要一次性设置，然后 plt.show() 就会弹出交互窗口
plt.rcParams['figure.figsize'] = (6, 4)   # 可选：图的大小
plt.rcParams['figure.dpi'] = 120          # 可选：分辨率

# 等价于 %config InlineBackend.figure_format = 'svg'
# 脚本里不需要，因为不会“嵌入”笔记本；若想存矢量图：
plt.savefig('myplot.svg', format='svg')   # 生成矢量文件

from IPython.core.interactiveshell import InteractiveShell # jupyter运行输出的模块
#显示每一个运行结果
InteractiveShell.ast_node_interactivity = 'all'
from IPython.core.interactiveshell import InteractiveShell # jupyter运行输出的模块
#显示每一个运行结果
InteractiveShell.ast_node_interactivity = 'all'

#设置行不限制数量
#pd.set_option('display.max_rows',None)

#设置列不限制数量
pd.set_option('display.max_columns', None)
data = pd.read_csv('datasets/000001.csv')
data['Day'] = pd.to_datetime(data['Day'],format='%Y/%m/%d')
data.set_index('Day', inplace = True)
data.sort_values(by = ['Day'], ascending=True)

data_new = data['1995-01':'2024-07'].copy()
data_new['Close'] = pd.to_numeric(data_new['Close'])
data_new['Preclose'] = pd.to_numeric(data_new['Preclose'])
# 计算000001上证指数日收益率 两种：
data_new['Raw_return'] = data_new['Close'] / data_new['Preclose'] - 1
print(data_new)

Month_data = data_new.resample('ME')['Raw_return'].apply(lambda x: (1+x).prod()-1).to_frame()
print(Month_data)

Month_data = data_new.resample('ME')['Raw_return'].apply(lambda x: np.prod(1+x)-1).to_frame()
print(Month_data)

Quarter_data = data_new.resample('QE')['Raw_return'].apply(lambda x: np.prod(1+x)-1).to_frame()
print(Quarter_data)

Year_data = data_new.resample('YE')['Raw_return'].apply(lambda x: np.prod(1+x)-1).to_frame()
print(Year_data)

# 更换列名字
Month_data.columns = ['Return']
print(Month_data)


import matplotlib.pyplot as plt
import matplotlib as mpl

#运行代码时候发现会报错，经过询问AI后得知缺乏相关字体，因此使用本机已经拥有的字体
# 查看本机有哪些可用字体
print([f.name for f in mpl.font_manager.fontManager.ttflist if 'Uni' in f.name or 'Sim' in f.name])

# 挑一个支持中文的
plt.rcParams['font.family'] = 'SimHei'

# 负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 画图
fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    'Return',  # 要画图的变量名
    '.-',  # 线的类型
    color = '#FFC0CB',  # 线的颜色 RGB
    label = 'Return',  # 这个是线的类别，主要是在多条线画图的时候，起到区别的作用，单条线这个没有影响
    linewidth = 1.5,  # 线的粗细
    data = Month_data['1995-01-01':'2024-07-31'])  # 画图的数据
ax.set_title("中国股票市场收益率 China's Stock Market") # 画图的标题
ax.set_xlabel('month') # 画图的x轴名称
plt.ylabel('Return') # 画图的y轴名称

# 设置x轴的日期格式
date_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_format)

# # 设置x轴的刻度间隔
ax.xaxis.set_major_locator(mdates.YearLocator())

# 旋转x轴标签以防止重叠
plt.xticks(rotation=90)

# 添加图例
plt.legend(loc='upper left', frameon=False, fontsize=10)
fig.savefig('Monthly_return.pdf', bbox_inches='tight')# 更改输出图片格式 jpg
plt.show();

help(plt.plot)

# 画图
fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    'Raw_return',  # 要画图的变量名
    '.-',  # 线的类型
    color = '#FFC0CB',  # 线的颜色 RGB
    label = 'Return',  # 这个是线的类别，主要是在多条线画图的时候，起到区别的作用，单条线这个没有影响
    linewidth = 1.5,  # 线的粗细
    data = Quarter_data['1995-01-01':'2024-07-31'])  # 画图的数据
ax.set_title("China's Stock Market") # 画图的标题
ax.set_xlabel('month') # 画图的x轴名称
plt.ylabel('Return') # 画图的y轴名称

# 设置x轴的日期格式
date_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_format)

# # 设置x轴的刻度间隔
ax.xaxis.set_major_locator(mdates.YearLocator())

# 旋转x轴标签以防止重叠
plt.xticks(rotation=90)

# 添加图例
plt.legend(loc='upper left', frameon=False, fontsize=10)
fig.savefig('Quaterly_return.pdf', bbox_inches='tight')# 更改输出图片格式 jpg
plt.show();

# 画图
fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    'Raw_return',  # 要画图的变量名
    '.-',  # 线的类型
    color = '#FFC0CB',  # 线的颜色 RGB
    label = 'Return',  # 这个是线的类别，主要是在多条线画图的时候，起到区别的作用，单条线这个没有影响
    linewidth = 1.5,  # 线的粗细
    data = Year_data['1995-01-01':'2024-07-31'])  # 画图的数据
ax.set_title("China's Stock Market") # 画图的标题
ax.set_xlabel('month') # 画图的x轴名称
plt.ylabel('Return') # 画图的y轴名称

# 设置x轴的日期格式
date_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_format)

# # 设置x轴的刻度间隔
ax.xaxis.set_major_locator(mdates.YearLocator())

# 旋转x轴标签以防止重叠
plt.xticks(rotation=90)

# 添加图例
plt.legend(loc='upper left', frameon=False, fontsize=10)
fig.savefig('Yearly_return.pdf', bbox_inches='tight')# 更改输出图片格式 jpg
plt.show();

# 画图
fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    'Raw_return',  # 要画图的变量名
    '.-',  # 线的类型
    color = '#FFC0CB',  # 线的颜色 RGB
    label = 'Return',  # 这个是线的类别，主要是在多条线画图的时候，起到区别的作用，单条线这个没有影响
    linewidth = 1.5,  # 线的粗细
    data = data_new['1995-01-01':'2024-07-31'])  # 画图的数据
ax.set_title("China's Stock Market") # 画图的标题
ax.set_xlabel('month') # 画图的x轴名称
plt.ylabel('Return') # 画图的y轴名称

# 设置x轴的日期格式
date_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_format)

# # 设置x轴的刻度间隔
ax.xaxis.set_major_locator(mdates.YearLocator())

# 旋转x轴标签以防止重叠
plt.xticks(rotation=90)

# 添加图例
plt.legend(loc='upper left', frameon=False, fontsize=10)
fig.savefig('Daily_return.pdf', bbox_inches='tight')# 更改输出图片格式 jpg
plt.show();