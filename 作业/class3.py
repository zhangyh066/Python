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

# 为月度收益创建辅助字段，便于深入分析
Month_data['Year'] = Month_data.index.year
Month_data['Month'] = Month_data.index.month
Month_data['Month_name'] = Month_data.index.strftime('%b')
Month_data['Decade'] = (Month_data['Year'] // 10) * 10
Month_data['Rolling_ann_return'] = (1 + Month_data['Ret']).rolling(window=12, min_periods=12).apply(lambda x: np.prod(x) - 1, raw=True)
Month_data['Rolling_ann_vol'] = Month_data['Ret'].rolling(window=12, min_periods=12).std() * np.sqrt(12)
Month_data

import matplotlib.pyplot as plt
import matplotlib as mpl

#运行代码时候发现会报错，经过询问AI后得知缺乏相关字体，因此使用本机已经拥有的字体
# 查看本机有哪些可用字体
print([f.name for f in mpl.font_manager.fontManager.ttflist if 'Uni' in f.name or 'Sim' in f.name])

# 挑一个支持中文的
plt.rcParams['font.family'] = 'SimHei'

# 负号正常显示
plt.rcParams['axes.unicode_minus'] = False


fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    'Ret',  # 使用正确的列名 Ret
    '.-',  # 线的类型
    color = "#FFC0CB",  # 线的颜色 RGB
    label = 'Monthly Return',  # 更改为中文标签
    linewidth = 1.5,  # 线的粗细
    data = Month_data['1996-01-01':'2023-07-31'])  # 画图的数据
ax.set_title("China's Stock Market Return (1995-2024)") # 更简洁的标题
ax.set_xlabel('Date') # 将x轴名称改为中文
plt.ylabel('Return') # 将y轴名称改为中文

# 设置x轴的日期格式
date_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_format)

# 设置x轴的刻度间隔
ax.xaxis.set_major_locator(mdates.YearLocator(1))  # 每1年显示一个刻度

# 添加百分比格式
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# 旋转x轴标签以防止重叠
plt.xticks(rotation=90)

# 添加图例
plt.legend(loc='upper right', frameon=False, fontsize=10)

# 图例添加在具体的位置
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False, fontsize=10)

# 添加水平参考线表示零收益
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

# 保存图片
# 使用命令创建文件夹
import os
os.makedirs('images', exist_ok=True)
fig.savefig('images/Monthly_return.pdf', bbox_inches='tight')
plt.show();

help(plt.plot)

# 画图季度数据
# 画图
fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    Quarter_data.index,  # x轴数据
    Quarter_data['Ret'],  # y轴数据
    '.-',  # 线的类型
    color = "#5426B6",  # 线的颜色 RGB
    label = 'Quarterly Return',  # 更改为季度收益率标签
    linewidth = 1.5,  # 线的粗细
)
ax.set_title("China's Stock Market Quarterly Returns (1996-2023)") # 更精确的标题
ax.set_xlabel('Date') # 更改x轴名称
plt.ylabel('Return') # y轴名称

# 设置x轴的日期格式
date_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_format)

# 设置x轴的刻度间隔
ax.xaxis.set_major_locator(mdates.YearLocator(1))  # 每1年显示一个刻度

# 旋转x轴标签以防止重叠
plt.xticks(rotation=90)

# 添加百分比格式
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# 添加水平参考线表示零收益
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

# 添加图例
plt.legend(loc='upper left', frameon=False, fontsize=10)

fig.savefig('images/Quarterly_return.pdf', bbox_inches='tight')
plt.show();

# 画图
fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    Year_data.index,  # x轴数据
    Year_data['Ret'],  # y轴数据
    'o-',  # 线的类型，使用圆点标记每年的数据点
    color = '#4FCD21',  # 线的颜色，使用绿色以区分月度和季度图
    label = 'Annual Return',  # 更改标签为年度收益
    linewidth = 1.5,  # 线的粗细，略微加粗以突出年度数据
)
ax.set_title("China's Stock Market Annual Returns (1996-2023)") # 更精确的标题
ax.set_xlabel('Year') # 更改x轴名称
plt.ylabel('Return') # y轴名称

# 设置x轴的日期格式
date_format = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_format)

# 设置x轴的刻度间隔
ax.xaxis.set_major_locator(mdates.YearLocator(1))  # 每1年显示一个刻度

# 旋转x轴标签以防止重叠
plt.xticks(rotation=90)

# 添加百分比格式
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# 添加水平参考线表示零收益
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

# 为正负收益添加不同颜色的填充
for i, year in enumerate(Year_data.index):
    if Year_data['Ret'].iloc[i] >= 0:
        ax.bar(year, Year_data['Ret'].iloc[i], width=200, alpha=0.3, color='red')
    else:
        ax.bar(year, Year_data['Ret'].iloc[i], width=200, alpha=0.3, color='green')

# 添加图例
plt.legend(loc='upper left', frameon=False, fontsize=10)

# 保存图片到images文件夹
fig.savefig('images/Yearly_return.pdf', bbox_inches='tight')
plt.show();


# 添加图例
plt.legend(loc='upper left', frameon=False, fontsize=10)
fig.savefig('Daily_return.pdf', bbox_inches='tight')# 更改输出图片格式 jpg
plt.show();

data_new['Cum_return'] = (1 + data_new['Raw_return']).cumprod() - 1

# Cumulative return of Shanghai Index (base=0)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_new.index, data_new['Cum_return'], color='#ED73BA', linewidth=1.2, label='Cumulative Return')
ax.fill_between(data_new.index, 0, data_new['Cum_return'], color="#1F037D", alpha=0.15)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Shanghai Index Cumulative Return (1996-2023)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(rotation=90)
plt.legend(frameon=False) # 图例不显示边框
fig.savefig('images/Cumulative_return.pdf', bbox_inches='tight')
plt.show();

# 画图
fig, ax = plt.subplots(figsize=(10, 5)) # 图片比例
ax.plot(
    'Raw_return',  # 要画图的变量名
    '.-',  # 线的类型
    color = '#73BAED',  # 线的颜色 RGB
    label = 'Return',  # 这个是线的类别，主要是在多条线画图的时候，起到区别的作用，单条线这个没有影响
    linewidth = 1.5,  # 线的粗细
    data = data_new['1996-01-01':'2023-07-31'])  # 画图的数据
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
fig.savefig('images/Daily_return.pdf', bbox_inches='tight')# 更改输出图片格式 jpg
plt.show();

data_new['Cum_return'] = (1 + data_new['Raw_return']).cumprod() - 1

# Cumulative return of Shanghai Index (base=0)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_new.index, data_new['Cum_return'], color='#4876FF', linewidth=1.2, label='Cumulative Return')
ax.fill_between(data_new.index, 0, data_new['Cum_return'], color="#1F037D", alpha=0.15)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Shanghai Index Cumulative Return (1996-2023)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(rotation=90)
plt.legend(frameon=False) # 图例不显示边框
fig.savefig('images/Cumulative_return.pdf', bbox_inches='tight')
plt.show();

# Calculate drawdown
data_new['Peak'] = data_new['Cum_return'].cummax() # 计算历史最高点
data_new['Drawdown'] = data_new['Cum_return'] - data_new['Peak'] # 计算回撤

# 最大回撤曲线，关注下行风险
fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(data_new.index, data_new['Drawdown'], 0, color='#EDDC73', alpha=0.6, label='Drawdown')
ax.set_title('Shanghai Index Maximum Drawdown (1996-2023)')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(data_new['Drawdown'].min() * 1.05, 0.02)
plt.xticks(rotation=45)
plt.legend(frameon=False)
fig.savefig('images/Maximum_drawdown.pdf', bbox_inches='tight')
plt.show();

# 月度收益率分布：看集中度与尾部风险
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(Month_data['Ret'].dropna(), bins=40, kde=True, color='#4876FF', ax=ax)
mean_return = Month_data['Ret'].mean()
median_return = Month_data['Ret'].median()
ax.axvline(mean_return, color='#FF6A6A', linestyle='--', linewidth=1.2, label=f"Mean: {mean_return:.2%}")
ax.axvline(median_return, color='#2E8B57', linestyle='-.', linewidth=1.2, label=f"Median: {median_return:.2%}")
ax.set_title("Monthly Return Distribution (1996-2023)")
ax.set_xlabel('Monthly Return')
ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0)) # x轴百分比格式
plt.legend(frameon=False)
fig.savefig('images/Monthly_return_distribution.pdf', bbox_inches='tight')
plt.show();

# Boxplot: Monthly return distribution by month
month_order = list(calendar.month_abbr[1:])
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=Month_data.dropna(subset=['Ret']), x='Month_name', y='Ret', hue='Month_name', order=month_order, palette='Blues', ax=ax, legend=False)  # hue 表示每个箱线图根据月份着色，进一步区分不同月份。palette='Blues' 选择蓝色系配色方案，使不同月份的箱线图颜色有层次变化。
ax.set_title('Monthly Return Distribution by Month (1996-2023)')
ax.set_xlabel('Month')
ax.set_ylabel('Monthly Return')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)  # Add reference line at zero
plt.xticks(rotation=0)
fig.savefig('images/Monthly_return_boxplot.pdf', bbox_inches='tight')
plt.show();

#%%
# Create a year-month pivot table for the heatmap
heatmap_data = Month_data.pivot_table(values='Ret', index='Year', columns='Month', aggfunc='first').sort_index()
heatmap_data.columns = [calendar.month_abbr[m] for m in heatmap_data.columns]

# Create heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data * 100,
            cmap='RdYlGn_r',  # Red (negative) to Green (positive) colormap
            center=0,       # Center the colormap at zero
            linewidths=0.4,
            linecolor='white',
            cbar_kws={'label': 'Monthly Return (%)'},
            annot=True,     # Show values in cells
            fmt='.1f',      # Format as 1 decimal place
            ax=ax)

ax.set_title('Monthly Returns Heatmap (%)')
ax.set_xlabel('Month')
ax.set_ylabel('Year')

plt.tight_layout()
fig.savefig('images/Monthly_heatmap.pdf', bbox_inches='tight')
plt.show();

from statsmodels.tsa.stattools import acf

# 计算自相关函数
lags = 20
autocorr = acf(Month_data['Ret'].dropna(), nlags=lags, fft=True)

# 删除第一个数据点（lag=0，值为1）
autocorr_no_zero = autocorr[1:]

# 绘制自相关图
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, lags+1), autocorr_no_zero, alpha=0.7, color='#4876FF')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axhline(y=1.96/np.sqrt(len(Month_data['Ret'].dropna())), color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=-1.96/np.sqrt(len(Month_data['Ret'].dropna())), color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('Monthly Return Autocorrelation Function')
ax.set_xlabel('Lag (months)')
ax.set_ylabel('Autocorrelation')
ax.grid(True, alpha=0.3)

fig.savefig('images/Return_autocorrelation.pdf', bbox_inches='tight')
plt.show();

from statsmodels.tsa.stattools import acf

# 日数据的自相关的图

lags = 20
daily_autocorr = acf(data_new['Raw_return'].dropna(), nlags=lags, fft=True)

# 删除第一个数据点（lag=0，值为1）
daily_autocorr_no_zero = daily_autocorr[1:]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, lags+1), daily_autocorr_no_zero, alpha=0.7, color='#4876FF')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axhline(y=1.96/np.sqrt(len(data_new['Raw_return'].dropna())), color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=-1.96/np.sqrt(len(data_new['Raw_return'].dropna())), color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('Daily Return Autocorrelation Function')
ax.set_xlabel('Lag (days)')
ax.set_ylabel('Autocorrelation')
ax.grid(True, alpha=0.3) # 使网格线更淡一些
fig.savefig('images/Daily_return_autocorrelation.pdf', bbox_inches='tight')
plt.show();

# 按年代分析收益率分布
fig, axes = plt.subplots(2, 2, figsize=(8, 5))
decades = Month_data['Decade'].unique()
decades = sorted([d for d in decades if not pd.isna(d)])

for i, decade in enumerate(decades):
    row = i // 2
    col = i % 2
    decade_data = Month_data[Month_data['Decade'] == decade]['Ret'].dropna()

    axes[row, col].hist(decade_data, bins=20, alpha=0.7, color=f'C{i}', edgecolor='black', linewidth=0.5)
    axes[row, col].axvline(decade_data.mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {decade_data.mean():.2%}')
    axes[row, col].set_title(f'{int(decade)}s Monthly Returns')
    axes[row, col].set_xlabel('Monthly Return')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('images/Decades_return_distribution.pdf', bbox_inches='tight')
plt.show();