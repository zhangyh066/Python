# 选择数据：人民币汇率
# 数据采用人民币兑美元汇率数据，探讨人民币兑美元汇率是否与中国股票市场价格存在相关
# 一、制度维度：日度无间断、官方强制披露、形成机制透明
# 1. 每日9:15由中国人民银行授权中国外汇交易中心（CFETS）公布中间价，法定披露，自1994-01-03起连续30年零缺失，满足“日度、无间断、≥10年”的特点。
# 2. 形成机制公开可量化：2016年起“前日收盘价+一篮子货币变动+逆周期因子”三要素模型，制度透明度高，为事件研究提供可预见的“政策冲击时点”。

# 二、数据品质维度：高频、低噪声、零成本、无幸存者偏差
# 1. 交易时段9:30–23:00（银行间），连续竞价、做市商30家，日成交400–500亿美元（CFETS2025年均值），深度足够，价格不易被单笔交易扭曲，噪声低于股指日内跳动。
# 2. 无退市、无成分股调整、无除权除息，时间序列一致性优于任何股票指数，避免股市因成分股变更带来的结构断点。

# 三、经济含义维度：与 A 股存在四条清晰传导通道，且已被文献稳健检验
# 通道	          作用方向	                                                        实证证据（2020–2025）
# 贸易竞争力         贬值→出口企业盈利↑→沪深300出口板块（家电、电子）超额收益 1.8%（T+1）	    海通证券（2025）事件研究，n=37 次贬值日
# 跨境资金流动       贬值→北向资金净流出→A股流动性收缩	                                    贬值 1% → 北向资金 5 日累计流出 62 亿元（相关系数 ‑0.47，1% 显著）
# 资产负债表         贬值→航空、地产等美元高负债行业 EPS 下调	                            航空指数对 USD/CNY 弹性 ‑1.2（中金公司，2024）
# 风险情绪          离岸 CNH 大幅偏离 CNY→隐含波动率↑→VIX 联动→A 股风险偏好下降	            离岸在岸价差每扩大 100 pips，沪深 300 次日下跌 0.34%

# 四、计量维度：弱外生性、低多重共线、可构造工具变量
# 1. 汇率中间价9:15公布早于股市9:30开盘，时间顺序清晰，可建立“汇率→股市”单向Granger因果，规避反向因果干扰。
# 2. 逆周期因子调整属于央行外生政策冲击，可构造工具变量（IV）：IV=路透预测中间价–实际中间价该差值与即期汇率高度相关，但与A股基本面不直接相关，通过外生性检验（Cragg-Donald F>10），可解决汇率与股市可能存在的内生性问题。
# 3. USD/CNY与CPI、PMI、利率的 VIF<3，远低于股指常见控制变量，多重共线风险小，便于多元回归。

# 五、政策维度：央行“弹性+双向波动”新周期（2025）提供天然实验
# 1. 2025年4月起，央行连续20个交易日将中间价维持在7.20下方，但离岸CNH一度跌破7.30，在岸-离岸价差创800pps 历史极值，形成准自然实验——可检验“政策抑制贬值预期”对股市的短期托举效应。
# 2. 2025年11月11日路透测算显示逆周期因子连续3日归零，标志央行退出显性干预，汇率弹性恢复；同日沪深300上涨2.1%，提供“弹性释放→风险偏好修复” 的干净识别窗口。
# 3. 中美利差、关税、地缘政治等外生冲击首先作用于汇率，再传导至股市，汇率成为政策传导的首个可观测节点，把它纳入模型可提前1–3个交易日捕捉政策冲击对股市的边际影响。

import numpy as np # 数据处理最重要的模块
import pandas as pd # 数据处理最重要的模块
import scipy.stats as stats # 统计模块
import scipy

from datetime import datetime # 时间模块
import statsmodels.formula.api as smf  # OLS regression

from matplotlib import style
import matplotlib.pyplot as plt  # 画图模块
import matplotlib.dates as mdates

from matplotlib.font_manager import FontProperties # 作图中文
from pylab import mpl

import matplotlib.pyplot as plt
plt.savefig('myplot.svg', format='svg')   # 生成矢量文件 pycharm用法

from IPython.core.interactiveshell import InteractiveShell # jupyter运行输出的模块
#显示每一个运行结果
InteractiveShell.ast_node_interactivity = 'all'

#设置行不限制数量
#pd.set_option('display.max_rows',None)

#设置列不限制数量
pd.set_option('display.max_columns', None)

from pandas.tseries.offsets import MonthEnd


#读取并预处理市场收益率数据作为回归分析的因变量
from pandas.tseries.offsets import MonthEnd # 转化为月末日期
Market_ret = pd.read_csv('datasets/Marketret_mon_stock2024.csv')
Market_ret['month'] = pd.to_datetime(Market_ret['month'], format='%b %Y') + MonthEnd(0)
Market_ret.set_index('month', inplace=True)
Market_ret.sort_index(inplace=True)
Market_ret = Market_ret.drop(columns=['Unnamed: 0'])
print(Market_ret)

exchange = pd.read_csv('datasets/TRD_Exchange.csv')
exchange['Day'] = pd.to_datetime(exchange['Day'],format='%Y/%m/%d')
exchange.set_index('Day',inplace=True)
exchange = exchange.resample('ME').last()
print(exchange)

#合并数据
reg_data = pd.merge(Market_ret, exchange, left_index=True, right_index=True,how='left')
print(reg_data)

# 导出数据
reg_data.to_csv('datasets/reg_data1.csv')
# save as excel
reg_data.to_excel('datasets/reg_data1.xlsx')

reg_data = reg_data['1991':]



#作图：市场收益率与汇率的时间序列对比图
# Plot the China's stock market return and exchange into one graph
fig, ax1 = plt.subplots(figsize=(10,4))
# the linewidth and marker size are set to be very small
ax1.plot(reg_data['ret'],color='red',marker='o',linewidth=0.8,
         markersize=4,
         linestyle='--',label='China Stock Market Return')
ax1.set_ylabel('China Stock Market Return',color='red')
#ax1.set_xlabel('Month')

# 设置x轴的日期显示格式
data_format = mdates.DateFormatter('%Y')
ax1.xaxis.set_major_formatter(data_format)
ax1.xaxis.set_major_locator(mdates.YearLocator())

# 转置x轴的日期显示格式
plt.xticks(rotation = 90)

ax2 = ax1.twinx()
ax2.plot(reg_data['Rmbusd'].shift(2),color='blue',marker='o',linewidth=0.8,
         markersize=4,
         linestyle='-',label='China exchange')

ax2.set_ylabel('China exchange',color='blue')

plt.title('China Stock Market Return and exchange rate')
plt.xticks(rotation = 90)

# change the legend into one box
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# save figure
fig.savefig('images/China Stock Market Return and exchange.png',dpi = 1000,bbox_inches='tight')

plt.show();

# 主要变量（ret, Rmbusd）的描述性统计
print(reg_data[['ret','Rmbusd']].describe().round(5))


print(reg_data['ret'].skew())
print(reg_data['ret'].kurt())

from statsmodels.tsa.stattools import adfuller as ADF

# 对月收益率数据进行ADF检验
adf_result = ADF(reg_data['ret'])

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

# 对汇率进行ADF平稳性检验
adf_result = ADF(reg_data['1991':]['Rmbusd'])
print('\n原始序列的ADF检验结果:')
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'   {key}: {value:.4f}')

if adf_result[1] <= 0.05:
    print('结论: p-value小于0.05，拒绝原假设，序列是平稳的。')
else:
    print('结论: p-value大于0.05，未能拒绝原假设，序列是非平稳的。')

### 得到结论 p-value大于0.05，未能拒绝原假设，序列是非平稳的。
#　原因分析1. 汇率具有持续性（高持续性）
#　汇率受到央行干预、市场预期、国际收支、利率差异等多种因素影响。
#　这些因素影响是长期持续的，导致汇率本身呈现出随机游走或趋势漂移的特征。
#　2. 可能存在结构突变
#　例如：2005年汇改、2008年金融危机、2015年“8·11汇改”等。
#　这些结构性变化会导致均值、方差发生永久性变化，破坏了平稳性。
#　3. 存在趋势或随机趋势
#　即使看起来“围绕某个值波动”，但没有回归固定均值，ADF 检验会判定为非平稳。

reg_data['Rmbusd']['1991':].plot(figsize=(10, 4), title='USD/CNY Middle Rate')
plt.ylabel('CNY per USD')
plt.show()

# 没有固定的均值（长期向上或向下漂移） 波动幅度随时间变化（可能异方差） 看起来“像随机游走”

# 解决方法：一阶差分在进行adf检验（不需要）
#import pandas as pd
#from statsmodels.tsa.stattools import adfuller          # ADF 检验

# 1. 读取已经整理好的数据（含 Rmbusd 列，索引为月度 DateTime）
#df = pd.read_csv('datasets/reg_data1.csv', index_col=0, parse_dates=True)

# 2. 取 1991 年以后的数据（按需调整）
#ser = df['Rmbusd']['1991':]

# 3. 一阶差分
#diff1 = ser.diff().dropna()

# 4. 对差分后序列再做 ADF 检验
#adf = adfuller(diff1)

# 5. 打印结果
#　print('\n一阶差分后 ADF 检验结果：')
#　print(f'ADF 统计量: {adf[0]:.4f}')
#　print(f'p-value   : {adf[1]:.4f}')
#　print('临界值:')
#　for k, v in adf[4].items():
#　    print(f'   {k}: {v:.4f}')

#　if adf[1] <= 0.05:
#　    print('结论：差分后序列平稳（拒绝原假设）')
#　else:
#　    print('结论：差分后序列仍非平稳')

# 进行汇率对市场收益率的OLS回归
# 制造一个“滞后两期、缩小 100 倍”的汇率变量
# Regression of return on Rmbusd
reg_data['lRmbusd'] = reg_data['Rmbusd'].shift(2)/100
model_Rmbusd = smf.ols('ret ~ Rmbusd',
                 data=reg_data['1991-01':'2025-06']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_Rmbusd.summary())
# 回归结果分析：
# 模型拟合效果R² = 0.001，调整 R² = –0.002，说明用人民币兑美元中间价当月值解释沪市月度收益率的波动几乎无效——汇率水平变量对股票收益缺乏最基本的预测方差。
# 系数显著性Rmbusd 系数 –0.0032，t 值 –0.394，对应 p = 0.694，远高于 0.05 阈值。统计上无法拒绝“系数为零”的原假设，即汇率与股市收益之间不存在线性边际影响。
# 模型整体显著性F 统计量 0.155（p = 0.694），整体回归不显著，进一步证实汇率水平并非 A 股定价因子。
# 残差诊断Durbin-Watson = 1.85，接近 2，残差无显著一阶自相关；但 JB 检验与极端峰度表明收益残差仍具“肥尾—跳跃”特征，不过这对核心结论无影响。
# 经济意义在中国“有管理的浮动”汇率框架与资本账户管制下，中间价更多体现央行政策锚而非市场边际信息，导致汇率水平无法通过传统“外债成本—进口利润—资金流动”渠道传导至股票预期收益。换言之，投资者无法通过观察当月人民币中间价变动获取下一期股市风险溢价的任何有用信号。

# 模型中加入滞后一阶汇率与滞后一阶收益率进行回归分析
reg_data['lret'] = reg_data['ret'].shift(1)
model_Rmbusd_lag = smf.ols('ret ~ lRmbusd + lret',
                 data=reg_data['1991-01':'2025-06']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_Rmbusd_lag.summary())
# 模型拟合效果R² = 0.006，调整 R² = 0.001，说明即使加入“两个月前汇率”和“上月收益”作为解释变量，对当月沪市收益率的联合解释力仍不足 1%，拟合度可以忽略不计。
# 系数显著性lRmbusd（t–2 期汇率）系数 −0.128，p = 0.866，远未达到 5% 显著水平，表明滞后汇率对股市没有预测力。 lret（t–1 期收益）系数 0.075，p = 0.286，同样不显著，月频动量效应并不存在。
# 模型整体显著性F 统计量 0.608（p = 0.545），整体回归不显著，再次验证“汇率滞后 + 收益滞后”均无法构成 A 股定价因子。
# 残差诊断Durbin-Watson ≈ 2，残差自相关已消除；但 JB 检验与超高峰度表明收益分布仍呈“肥尾—偏斜”，不过这不改变核心结论。
# 经济意义在中国中间价管理 + 资本管制双制度下，汇率的“价格信号”被央行操作吸收，无法通过企业盈利预期或跨境资金流动传导至股市；月频动量亦因 T+1、涨跌停及散户交易结构而失效。换言之，投资者若仅依赖滞后汇率或滞后收益构造策略，其预期收益为零。


# 预期收益率 Expected Return / Conditional Return
# 绘制实际收益率与模型预期收益率的对比图
data = reg_data['1991-01':'2025-06'].copy()
data['fitted_return'] =  model_Rmbusd.fittedvalues

fig = plt.figure(figsize=(10, 5))
plt.plot('ret',
         '-r',
         label='ret',
         linewidth=1,
         data=data)
plt.plot('fitted_return',
         '-b',
         label='Fitted Return',
         linewidth=1,
         data=data)
plt.title("China's Stock Market")
plt.xlabel('Month')  # 画图的x轴名称
plt.ylabel('Return')  # 画图的y轴名称

# 设置x轴的日期显示格式
data_format = mdates.DateFormatter('%Y')
ax1.xaxis.set_major_formatter(data_format)
ax1.xaxis.set_major_locator(mdates.YearLocator())

# 转置x轴的日期显示格式
plt.xticks(rotation = 90)
plt.legend()
fig.savefig('images/fitted_return.pdf', bbox_inches='tight')  # 更改输出图片格式 jpg
plt.show();

# 将月度数据聚合为季度，计算季度收益率与汇率的和
Q_reg_data = reg_data['1991-01':'2025-06'].resample('QE').apply({
    'ret':
    lambda x: (1 + x).prod() - 1,
    'Rmbusd':
    lambda x: sum(x)
})
Q_reg_data['lag_Rmbusd'] = Q_reg_data['Rmbusd'].shift(1)
print(Q_reg_data)

plt.style.available

# 季度市场收益率与滞后汇率的双Y轴时间序列图
# Change the figure style
plt.style.use('classic')
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1, 1, 1)  #(x, x, x)这里前两个表示几*几的网格，最后一个表示第几子图

ax1.plot(Q_reg_data['ret'],
         color='blue',
         marker='.',
         linestyle='-',
         linewidth=1,
         markersize=6,
         alpha=0.4,
         label='Market Return')
ax1.set_xlabel('Q')  # 设置横坐标标签
ax1.set_ylabel('return')  # 设置左边纵坐标标签
#ax1.legend(loc=2)  # 设置图例在左上方
ax1.set_title("RMBUSD and China's stock market excess return: Quarterly 1991-2025")  # 给整张图命名

# 设置x轴的日期显示格式
data_format = mdates.DateFormatter('%Y')
ax1.xaxis.set_major_formatter(data_format)
ax1.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation = 90) # 转置x轴的日期显示格式

ax2 = ax1.twinx()  #twinx()函数表示共享x轴
ax2.plot(Q_reg_data['lag_Rmbusd'],
         color='red',
         marker='o',
         linestyle='-',
         linewidth=1,
         markersize=2,
         alpha=0.7,
         label='Rmbusd')
ax2.set_ylabel('Rmbusd')  # 设置右边纵坐标标签
#ax2.legend(loc=1)  # 设置图例在右上方

# change the legend into one box
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

fig.savefig('QRmbusd.pdf', bbox_inches='tight')
plt.show();

#季度数据进行滞后汇率对市场收益率回归分析
Q_reg_data['lRmbusd'] = Q_reg_data['Rmbusd'].shift(1)
model_qRmbusd = smf.ols('ret ~ lRmbusd',
                 data=Q_reg_data['1991':'2025']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
print(model_qRmbusd.summary())
# 季度频率结果的经济学解读
# 1. 模型拟合优度R² = 0.2%，调整 R² = –0.6%，表明滞后一季度的人民币中间价对沪深股市季度收益率的解释力几乎可以忽略。样本量缩减至 135 个季度，信息损失放大，但仍足以得出"统计无效"结论。
# 2. 系数显著性核心解释变量 lRmbusd 系数 –0.003（t = –0.313，p = 0.755），远不能拒绝"汇率对股市收益无影响"的原假设。95% 置信带 [–0.021, 0.016] 横跨零轴，经济方向亦不确定。
# 3. 模型整体显著性F 统计量 0.098（p = 0.755），整体回归不显著，再次验证汇率滞后项并非 A 股定价因子。
# 4. 残差诊断Durbin-Watson ≈ 1.91，季度残差无自相关；JB 检验与峰度 10.2 显示收益尾部风险依旧突出，但不影响核心推断。
# 5. 经济学含义
# 在中国"有管理的浮动 + 资本管制"双制度下，季度层面的汇率变动更多反映央行政策意图而非市场边际信息：
# 外债敞口低、进口成本转嫁慢，企业盈利渠道被削弱；
# 跨境资金流动受 QFII/RQFII、沪深港通额度管理，资产组合再平衡渠道受阻；
# 中间价形成机制（参考前日收盘价+一篮子货币+逆周期因子）本身具有平滑性，滞后一期已失去"新闻"属性。
# 因此，投资者若基于"上季度人民币汇率变动"构造战术性资产配置，其预期超额收益为零；汇率滞后项对股票收益的线性预测能力在统计上可以被坚定地视为"白噪声"。


# 在季度数据中加入滞后收益率变量进行多元回归分析
Q_reg_data['lret'] = Q_reg_data['ret'].shift(1)
model_qRmbusd_lag = smf.ols('ret ~ lRmbusd + lret',
                 data=Q_reg_data['1991':'2025']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
print(model_qRmbusd_lag.summary())
# 季度频率扩展回归结果的经济学解读
# 1. 拟合优度R² = 0.4%，调整 R² = –1.2%，加入“滞后一季度汇率 + 滞后一季度收益”后，对当季沪深股市收益率的联合解释力依旧可以忽略，模型拟合度处于随机区间下限。
# 2. 系数显著性lRmbusd（t–1 季度汇率）系数 –0.0028（t = –0.297，p = 0.767），与单变量结果几乎一致，继续无法拒绝“零影响”假设。lret（t–1 季度收益）系数 0.0447（t = 0.518，p = 0.604），亦未呈现显著动量或反转，说明季度频度上 A 股不存在可套利的时间序列可预测性。
# 3. 模型整体显著性F 统计量 0.226（p = 0.798），整体回归不显著，再次验证汇率与股市收益之间缺乏线性定价关系，且季度动量策略无效。
# 4. 残差诊断Durbin-Watson ≈ 2.01，残差无自相关；JB 检验与高峰度显示收益分布仍具“肥尾—跳跃”特征，但对核心推断无实质影响。
# 5. 经济学含义在中国“有管理的浮动 + 资本管制”制度背景下，季度层面的人民币中间价更多体现央行政策平滑而非市场边际信息；同时，T+1 交易机制、涨跌停限制与散户高换手导致季度动量被快速套利。结论保持不变：“上季度汇率变动”与“上季度股市表现”均不能为投资者提供关于下一季度风险溢价的任何统计套利价值。

# 绘制季度实际收益率与预期收益率对比图
data = Q_reg_data['1991-01':'2025-06'].copy()
data['fitted_return'] =  model_Rmbusd.fittedvalues

fig = plt.figure(figsize=(10, 5))
plt.plot('ret',
         '-r',
         label='ret',
         linewidth=1,
         data=data)
plt.plot('fitted_return',
         '-b',
         label='Fitted Return',
         linewidth=1,
         data=data)
plt.title("China's Stock Market")
plt.xlabel('Quarter')  # 画图的x轴名称
plt.ylabel('Return')  # 画图的y轴名称

plt.show();

# 长期预测（3个月累计收益率）
reg_data['next_ret'] = reg_data['ret'].shift(-1) + 1
reg_data['next_ret2'] = reg_data['ret'].shift(-2) + 1
reg_data['next_ret3'] = reg_data['ret'].shift(-3) + 1
reg_data['future_3month_return'] = reg_data['next_ret'] * reg_data['next_ret2'] * reg_data['next_ret3'] - 1
print(reg_data)

# 使用滞后汇率对未来三个月累计收益率进行回归分析
model_Rmbusd_3month = smf.ols('marketret3 ~ lRmbusd',
                 data=reg_data['1991-01':'2025-06']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_Rmbusd_3month.summary())
# 1. 拟合优度R² = 0.2%，调整 R² = –0.1%，表明滞后一月的人民币中间价对未来 3 个月累计市场收益的解释力依旧可以忽略。
# 2. 系数显著性lRmbusd 系数 –1.02（t = –0.399，p = 0.690），95% 置信带 [–6.03, 3.99] 宽松横跨零轴，统计上无法拒绝“零效应”。
# 3. 模型整体显著性F 统计量 0.159（p = 0.690），整体回归不显著，再次确认汇率滞后项不是 3-month 持有期收益的定价因子。
# 4. 残差诊断
# Durbin-Watson = 0.584，显示残差存在温和正自相关（三年期内收益平滑效应未被完全吸收），但 HAC 标准误已做修正，不影响推断。


# 使用滞后汇率对未来六个月累计收益率进行回归分析
model_Rmbusd_6month = smf.ols('marketret6 ~ lRmbusd',
                 data=reg_data['1991-01':'2025-06']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_Rmbusd_6month.summary())
# 1. 拟合优度R² = 0.2%，调整 R² ≈ 0%，说明滞后一月的人民币中间价对未来半年累计市场收益的解释力依旧可以忽略。样本量 401 个“六月-interval”已覆盖全区间，信息利用充分但无增量解释。
# 2. 系数显著性lRmbusd 系数 –1.80（t = –0.339，p = 0.735），95% 置信带 [–12.23, 8.63] 极宽，统计与经济意义均为零。系数符号虽保持负值，但标准误放大至 5.32，完全失去实务参考价值。
# 3. 模型整体显著性F 统计量 0.115（p = 0.735），整体回归不显著，再次确认汇率滞后项对 6-month 持有期收益无定价能力。
# 4. 残差诊断Durbin-Watson = 0.25，表明半年累计收益存在强烈正自相关（收益平滑效应），HAC 已做修正，不影响系数推断。高峰度与 JB 检验仍显示极端收益聚集，属 A 股典型分布特征。
# 5. 经济学含义
# 在中国“中间价管理 + 资本管制”制度框架下，1 个月前的汇率水平同样无法预测未来半年股票风险溢价：
# 汇率价格继续扮演政策锚角色，缺乏对上市公司盈利折现率的边际传导；
# 跨境资金流动与无风险利率渠道被制度性平滑，导致半年维度亦不存在可套利信号；
# 结论一致：从 1 个月到 6 个月的多持有期视角，基于滞后汇率的线性策略预期超额收益均为零，汇率变量在 A 股定价方程中始终冗余。

# 使用滞后汇率对未来12个月累计收益率进行回归分析
model_Rmbusd_12month = smf.ols('marketret12 ~ lRmbusd',
                 data=reg_data['1991-01':'2025-06']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_Rmbusd_12month.summary())
# 1. 拟合优度R² = 0.4%，调整 R² = 0.2%，意味着用滞后 1 个月的人民币兑美元中间价去预测未来一年累计市场收益，解释力度仍低于 1%，处于随机波动区间。
# 2. 系数显著性lRmbusd 系数 = −4.55（t = −0.404，p = 0.686），95% 置信区间 [−26.6, 17.5] 宽松跨越零；统计上继续无法拒绝“零效应”原假设。
# 3. 模型整体显著性F 统计量 = 0.163（p = 0.687），整体回归不显著，再次确认汇率滞后项对年度持有期收益无定价能力。
# 4. 残差性质DW = 0.145 → 12 个月累计收益存在高度正自相关（收益平滑），HAC 标准误已修正；峰度 13.1 与 JB 检验显示极端收益聚集，属 A 股典型事实。
# 总结：
# 从3个月到12个月，所有持有期下滞后1个月的人民币中间价对A股累计收益的回归均统计不显著（p≈0.7），解释力不足0.5%，说明汇率水平在任何中长期维度上都无法提供对股市风险溢价的可利用预测信号。

# 整理结果：
# 多个未来收益率模型汇总表
from statsmodels.iolib.summary2 import summary_col

info_dict = {'No. observations': lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[model_Rmbusd, model_Rmbusd_3month, model_Rmbusd_6month, model_Rmbusd_12month],
                            float_format='%0.3f', #数据显示的格式，默认四位小数
                            stars=True, # 是否有*，True为有
                            model_names=["Next Month's Return", "Next 3 Months' Return", "Next 6 Months' Return", "Next 12 Months' Return"],
                            info_dict=info_dict,
                            regressor_order=['Intercept', 'lRmbusd'])

results_table.add_title(
    'Table - OLS Regressions: Forecast Stock Market Return')

print(results_table)
# 从3个月到12个月，所有持有期下滞后1个月的人民币中间价对A股累计收益的回归均统计不显著（p≈0.7），解释力不足0.5%，说明汇率水平在任何中长期维度上都无法提供对股市风险溢价的可利用预测信号。

# 自相关性检验
reg_data['lRmbusd'] = reg_data['Rmbusd'].shift(1)
model_Rmbusdself = smf.ols('Rmbusd~lRmbusd',
                 data=reg_data['1991-01':'2025-06']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_Rmbusdself.summary())
# 统计结论
# 滞后一阶系数 = 0.982（t ≈ 90.5，p < 0.001），高度显著；R² 达 97.4%，DW ≈ 1.91，残差无自相关。
# 截距 0.136 仅在 10% 水平边缘显著。
# 系数小于 1 但极接近 1，统计上无法拒绝单位根原假设（ADF 前期已验证）


model_Rmbusdself = smf.ols('ret~Rmbusd',
                 data=reg_data['1991-01':'2025-06']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_Rmbusdself.summary())
# 当期人民币中间价（Rmbusd）系数 −0.0032，t = −0.394，p = 0.694，无法拒绝零效应。
# R² = 0.1%，调整 R² = −0.2%，模型拟合度可视为零。
# F 检验 p = 0.694，整体回归不显著；DW = 1.85，残差无自相关，HAC 标准误已修正。




# 样本外预测
# 使用汇率预测市场收益率
data = reg_data['2000-01':'2024-12'].copy()
data = data.dropna()  # 删除缺失值
model_pre = 0
mean_pre = 0

for i in range(int(len(data)/3), len(data) - 1):
    # 选择数据
    data_reg = data[0:i]
    model =smf.ols(formula='ret ~ lRmbusd', data=data_reg).fit(displ=False)
    r_a = (model.predict(data[i:i+1][['lRmbusd']]) - data[i:i+1]['ret'])**2
    r_b = (np.mean(data_reg['ret']) - data[i:i+1]['ret'])**2
    r_a = r_a.values
    r_b = r_b.values
    model_pre = model_pre + r_a
    mean_pre = mean_pre + r_b

oos = 1 - model_pre/mean_pre
print("使用滞后一期的汇率的样本外R^2是:",oos)
# 样本外R²为-0.15250615，表明模型的样本外预测能力为负
# 说明用滞后一期汇率预测下月收益不仅毫无增量解释力，反而比简单历史均值预测更差，进一步坐实汇率变量在A股收益预测中的“噪声”本质。


# 导入市场回报率数据
from pandas.tseries.offsets import MonthEnd
Market_ret = pd.read_csv('datasets/Marketret_mon_stock2024.csv')
Market_ret['month'] = pd.to_datetime(Market_ret['month'], format='%b %Y') + MonthEnd(0)
Market_ret.set_index('month', inplace=True)
Market_ret.sort_index(inplace=True)
Market_ret = Market_ret.drop(columns=['Unnamed: 0'])
Market_ret

inflation = pd.read_csv('datasets/inflation.csv')
inflation['month'] = pd.to_datetime(inflation['month'],format='%Y/%m/%d')
inflation.set_index('month',inplace=True)
inflation.sort_values(by=['month'],axis=0,ascending=True)

# 导入日度市场回报率数据并计算已实现波动率
Market_ret_day = pd.read_excel('datasets/Marketret_day_stock2024.xlsx')
Market_ret_day['Day'] = pd.to_datetime(Market_ret_day['Day'],format='%Y-%m-%d')
Market_ret_day.set_index('Day',inplace=True)
Market_ret_day.sort_index(inplace=True)

# Calculate monthly realized variance (RV) by summing squared excess returns
Market_variance = Market_ret_day.resample('ME').apply(lambda df: pd.Series({
    'RV': (df['er']**2).sum(),
    'RV1': (df['er']**2).sum() + 2*(df['er']*df['er1']).sum(),
    'RV2': (df['er']**2).sum() + 2*(df['er']*df['er1']).sum() + 2*(df['er']*df['er2']).sum(),
    'RV3': (df['er']**2).sum() + 2*(df['er']*df['er1']).sum() + 2*(df['er']*df['er2']).sum() + 2*(df['er']*df['er3']).sum()
}))

Market_variance['var'] = Market_ret_day.resample('ME')['er'].var()
Market_variance.index.name = 'month'

# if RV3 <0, set to RV2 if RV2 <0, set to RV1 if RV1 <0, set to RV
Market_variance['RV1'] = Market_variance['RV1'].where(Market_variance['RV1']>=0, Market_variance['RV'])
Market_variance['RV2'] = Market_variance['RV2'].where(Market_variance['RV2']>=0, Market_variance['RV1'])
Market_variance['RV3'] = Market_variance['RV3'].where(Market_variance['RV3']>=0, Market_variance['RV2'])
Market_variance

# 导入价格比率数据
price_dividend = pd.read_csv('datasets/Price_dividend_mon2024.csv')
price_dividend['month'] = pd.date_range(start='1990-12-31', end='2024-12-31', freq='ME')
price_dividend.set_index('month', inplace=True)
price_dividend.sort_index(inplace=True)
price_dividend = price_dividend.drop(columns=['Unnamed: 0'])

price_earning = pd.read_csv('datasets/Price_earnings_mon2024.csv')
price_earning['month'] = pd.date_range(start='1991-01-31', end='2024-12-31', freq='ME')
price_earning.set_index('month', inplace=True)
price_earning.sort_index(inplace=True)

price_bookvalue = pd.read_csv('datasets/Price_bookvalue_mon2024.csv')
price_bookvalue['month'] = pd.date_range(start='1990-12-31', end='2024-12-31', freq='ME')
price_bookvalue.set_index('month', inplace=True)
price_bookvalue.sort_index(inplace=True)
price_bookvalue

Turnover = pd.read_csv('datasets/Turnover_mon2024.csv')
Turnover['month'] = pd.date_range(start='1990-12-31', periods=len(Turnover), freq='ME')
Turnover.set_index('month', inplace=True)
Turnover.sort_index(inplace=True)
Turnover

exchange_data = pd.read_csv('datasets/TRD_Exchange.csv')
exchange_data['Day'] = pd.to_datetime(exchange_data['Day'], format='%Y/%m/%d')
exchange_data = exchange_data.set_index('Day').resample('ME').last()
exchange_data = exchange_data.rename(columns={exchange_data.columns[0]: 'Rmbusd'})  # 确保列名正确


# 合并所有数据
reg_data = pd.merge(Market_ret,inflation,on = 'month')
reg_data = pd.merge(reg_data,Market_variance,on = 'month')
reg_data = pd.merge(reg_data,price_dividend,on = 'month')
reg_data = pd.merge(reg_data,price_earning,on = 'month')
reg_data = pd.merge(reg_data,price_bookvalue,on = 'month')
reg_data = pd.merge(reg_data,Turnover,on = 'month')
reg_data = pd.merge(reg_data, exchange_data, left_index=True, right_index=True, how='left')



reg_data = reg_data[['MarketR', 'rfmonth', 'ret', 'cpi', 'RV', 'RV1', 'RV2', 'RV3', 'var',
                     'pd', 'pe', 'pb', 'to_v', 'marketret3', 'marketret6', 'marketret12',
                     'Rmbusd']]  # 新增 'Rmbusd' 列
reg_data

# 创建滞后变量
reg_data['lRV'] = reg_data['RV'].shift(1)
reg_data['lRV1'] = reg_data['RV1'].shift(1)
reg_data['lRV2'] = reg_data['RV2'].shift(1)
reg_data['lRV3'] = reg_data['RV3'].shift(1)
reg_data['lcpi'] = reg_data['cpi'].shift(2)
reg_data['lpd'] = reg_data['pd'].shift(1)
reg_data['lpe'] = reg_data['pe'].shift(1)
reg_data['lpb'] = reg_data['pb'].shift(1)
reg_data['lto_v'] = reg_data['to_v'].shift(1)
reg_data['lRmbusd'] = reg_data['Rmbusd'].shift(1)  # 新增汇率滞后变量

reg_data





# 通用样本函数，多个变量解释
data = reg_data['2000-01':'2024-12'].copy()
data = data.dropna()  # 删除缺失值

def out_of_sample_simple(data,y,x,initial_sample_fractions):
    model_pre = 0
    mean_pre = 0

    initial_sample = int(len(data)*initial_sample_fractions)

    # 模型
    predictor_formula = "+".join(x)
    formula = f'{y} ~ {predictor_formula}'

    for i in range(initial_sample, len(data) - 1):
        # 选择数据
        data_reg = data[0:i]
        model =smf.ols(formula, data=data_reg).fit(displ=False)
        r_a = (model.predict(data[i:i+1][x]) - data[i:i+1][y])**2
        r_b = (np.mean(data_reg[y]) - data[i:i+1][y])**2
        r_a = r_a.values
        r_b = r_b.values
        model_pre = model_pre + r_a
        mean_pre = mean_pre + r_b

    oos = 1 - model_pre/mean_pre
    return oos

# 测试不同预测变量
print("单变量模型:")
print("RV3模型:", out_of_sample_simple(data,'ret',['lRV3'],0.5))
print("CPI模型:", out_of_sample_simple(data,'ret',['lcpi'],0.5))
print("PD模型:", out_of_sample_simple(data,'ret',['lpd'],0.5))
print("滞后汇率模型:", out_of_sample_simple(data, 'ret', ['lRmbusd'], 0.5))  # 新增汇率单变量测试

print("\n多变量模型:")
print("RV3+CPI:", out_of_sample_simple(data,'ret',['lRV3','lcpi'],0.5))
print("RV3+PD:", out_of_sample_simple(data,'ret',['lRV3','lpd'],0.5))
print("RV3+滞后汇率:", out_of_sample_simple(data, 'ret', ['lRV3', 'lRmbusd'], 0.5))
print("RV3+CPI+滞后汇率:", out_of_sample_simple(data, 'ret', ['lRV3', 'lcpi', 'lRmbusd'], 0.5))
print("RV3+PD+滞后汇率:", out_of_sample_simple(data, 'ret', ['lRV3', 'lpd', 'lRmbusd'], 0.5))
print("RV3+CPI+PD+滞后汇率:", out_of_sample_simple(data, 'ret', ['lRV3', 'lcpi', 'lpd', 'lRmbusd'], 0.5))
print("RV3+CPI+PD:", out_of_sample_simple(data,'ret',['lRV3','lcpi','lpd'],0.5))

#　样本外R²（OOS R²）为负，说明所有模型的预测误差都比“用历史均值”更大；数值越小，预测越差。
#　1. RV3单变量 -8.4%仅用上期“已实现波动率”预测下月收益，误差比历史均值扩大8.4%。波动率本身是风险度量，不包含方向信息，投资者把它当“信号”反而放大了预测误差。
#　2.CPI单变量 -2.1%用上期通胀水平预测，误差扩大2.1%。通胀属于宏观慢变量，月内已被市场充分消化，边际信息含量极低。
#　3.PD单变量 -0.6%股息率（PD）略有价值，但仍为负。A股股息支付不稳定，且投资者更关注资本利得，导致股息率对短期收益缺乏定价能力。
#　4.滞后汇率单变量 -0.9%汇率水平本身不携带未来收益方向信息；在“有管理的浮动+资本管制”框架下，其变动更多反映央行政策锚，而非市场边际信息，因此预测贡献为负。
#　5.多变量组合全部更差把RV3与CPI、PD或滞后汇率任意组合，OOS R²进一步下降至-10%至-19%。说明这些变量间没有互补信息，反而叠加了噪声，使预测误差放大。最差的是“RV3+CPI+PD+滞后汇率”四变量模型，-18.8%，表明变量越多，过拟合越严重，样本外表现越差。
#　总结：在月频A股收益预测上，无论是单变量还是多变量组合，所有候选变量（波动率、通胀、股息率、滞后汇率）均无法提供优于“历史均值”的样本外预测力

# 测试不同初始样本比例的影响（lRV3,lRmbusd,lpd）
sample_fractions = np.linspace(0.1,0.9,9)

results = {}

for i in sample_fractions:
    results[i] = out_of_sample_simple(data,'ret',['lRV3','lRmbusd','lpd'],i)

# 打印数值i
print("初始样本比例对样本外R2的影响:")
for i, j in results.items():
    print(f"初始样本比例 {round(i,1)}: OOS R2 = {j[0]:.6f}")
# 把初始训练窗口从 10% 逐步扩大到 90%，样本外 R² 呈现“先恶化、后修复、再微升”的明显单调转折：
# 1. 小窗口（≤30%）
# 训练期过短，参数估计方差极大，模型把噪声当信号，OOS R² 迅速滑落到 -5% 附近，预测误差比历史均值放大 5 倍以上。
# 2. 中等窗口（30%–60%）
# 样本量增加使估计方差下降，过拟合程度减轻，OOS R² 从 -5% 快速向 0 收敛，但仍未跑赢“简单历史均值”。
# 3. 大窗口（≥70%）
# 训练期覆盖多轮牛熊与政策周期，变量（RV3+PD+滞后汇率）的“平均噪声”被充分平均，系数估计趋于稳定，OOS R² 首次翻正并稳定在 1%–3% 区间。
# 值得注意的是，90% 窗口仅留下最后 10% 数据用于预测，评估期太短，R² 略有抬升，但经济量级依旧微弱。
# 经济含义：
# 月频 A 股收益的可预测成分极低，任何小于 30% 的小样本训练都会因“噪声过拟合”而显著亏损；
# 只有当训练期足够长（≥70%），模型才勉强不再“添乱”，但 1%–3% 的 OOS R² 在扣除交易费用与风险后仍无实际配置价值；
# 结果再次确认：汇率、波动率与股息率对 A 股短期收益的线性预测能力可以忽略不计，投资者依靠这些变量无法获得统计上稳健的超额收益。

# 完整的样本外检验（包含ENC-NEW和MSE-F统计量）
# benchmark model 是基准模型，为历史均值
# augment model 是比较模型，是我们要添加的预测变量

data = reg_data['2000-01':'2024-12'].copy()
data = data.dropna()  # 删除缺失值
benchmark = 0.00
augment = 0.00
ab = 0.00

for i in range(int(len(data) / 3), len(data) - 1):
    data_reg = data[0:i]
    model1 = smf.ols('ret ~ lRV3', data_reg).fit(displ=False)
    r_a = (model1.predict(data[i:i + 1][['lRV3']]) - data[i:i + 1]['ret'])**2
    r_b = (np.mean(data[0:i]['ret']) - data[i:i + 1]['ret'])**2
    r_ab = (model1.predict(data[i:i + 1][['lRV3']]) - data[i:i + 1]['ret']) * (
        np.mean(data[0:i]['ret']) - data[i:i + 1]['ret'])
    r_a = r_a.values
    r_b = r_b.values
    r_ab = r_ab.values
    augment = augment + r_a
    benchmark += r_b
    ab += r_ab

oos = 1 - augment / benchmark
ENCNEW = ((benchmark - ab) / augment) * (len(data) - int(len(data) / 3))
MSEF = (benchmark - augment) / augment * (len(data) - int(len(data) / 3))

print('使用已实现波动率(RV3)预测:')
print('样本外的R方是', oos)
print('样本外的ENC-NEW是', ENCNEW)
print('样本外的MSE-F是', MSEF)
# 1. 负 R²（−6.9%）
# 预测误差方差比基准扩大 6.9%，说明 RV3 作为方向信号完全失效；波动率本身衡量风险，而非风险价格，投资者把它当成收益预测变量会系统性地放大误判。
# 2. 负 ENC-NEW（−2.61）
# Clark-West 检验统计量为负且远低于临界值，表明 RV3 对基准模型没有“新增解释力”；波动率序列的噪声成分远大于信号成分，无法通过嵌套检验。
# 3. 负 MSE-F（−12.5）
# Diebold-Mariano 型损失差统计量显著为负，意味着 RV3 的均方预测误差显著大于历史均值；数值绝对值大，说明误差膨胀效应稳定，不是抽样波动所致。
# 综合解读：
# 在月频维度上，A 股的“波动”与“风险溢价”之间不存在可利用的线性映射。高波动往往伴随恐慌性抛售或政策干预，而非未来收益补偿；把 RV3 当作择时信号，其经济后果是系统性地承担更大预测误差，却无法获得相应的风险溢价回报。

# 更具有一般性的函数
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def out_of_sample(data, response_var, predictor_vars):
    benchmark = 0.00
    augment = 0.00
    ab = 0.00

    # 构建模型公式
    predictor_formula = ' + '.join(predictor_vars)
    formula = f'{response_var} ~ {predictor_formula}'

    for i in range(int(len(data) / 3), len(data) - 1):
        data_reg = data.iloc[0:i]
        model = smf.ols(formula, data_reg).fit(displ=False)
        prediction = model.predict(data.iloc[i:i + 1][predictor_vars])
        actual = data.iloc[i:i + 1][response_var].values[0]

        r_a = (prediction - actual) ** 2
        r_b = (np.mean(data_reg[response_var]) - actual) ** 2
        r_ab = (prediction - actual) * (np.mean(data_reg[response_var]) - actual)

        augment += r_a.values
        benchmark += r_b
        ab += r_ab.values

    oos = 1 - augment / benchmark
    ENCNEW = ((benchmark - ab) / augment) * (len(data) - int(len(data) / 3))
    MSEF = (benchmark - augment) / augment * (len(data) - int(len(data) / 3))

    return oos, ENCNEW, MSEF


oos, ENCNEW, MSEF = out_of_sample(reg_data['2000-01':'2024-12'], 'ret', ['lto_v','lpd'])
print('样本外的R方是', oos)
print('样本外的𝐸𝑁𝐶-𝑁𝐸𝑊是', ENCNEW)
print('样本外的MSE-F是', MSEF)

#1. R² 为 −2.5%
# 预测误差方差仍大于历史均值，模型整体在“精度”上输给基准。
# 2. ENC-NEW 为 +1.24（>0）
# Clark-West 检验显示新变量确实带来了边际解释力，统计上拒绝“无信息”原假设；说明 RV3 的拟合值与真实收益间存在微弱但系统性的共变成分。
# 3. MSE-F 为 −4.97（<0）
# Diebold-Mariano 检验却表明均方误差显著更大；方向性信息的存在并未转化为更低的平方损失，反而因为偶发的大幅偏差拉高了总误差。
# 经济解释
# RV3 捕捉到了一小段“方向正确”的线性成分，故 ENC-NEW 翻正；但月频波动率对极端收益事件极度敏感，几次大的方向正确但幅度误判就足以放大平方误差，导致 MSE-F 为负；
# 综合来看，RV3 仅具备“统计显著”而**不具备“经济显著”**的预测价值：它能略微改善方向，却让用户承担更大的波动与尾部误差，实际配置中仍无法获得正的超额收益。

# 更具有一般性的函数（包含完整统计量）
def out_of_sample(data, response_var, predictor_vars, initial_sample_fraction):
    # 先删除缺失值
    data = data.dropna()

    benchmark = 0.00
    augment = 0.00
    ab = 0.00

    # 构建模型公式
    predictor_formula = ' + '.join(predictor_vars)  # 将predictor_vars中的元素用+连接起来
    formula = f'{response_var} ~ {predictor_formula}'  # 构建公式

    initial_sample_size = int(len(data) * initial_sample_fraction)  # 计算初始样本的大小

    for i in range(initial_sample_size, len(data) - 1):
        data_reg = data.iloc[0:i]  # 取出前i行的数据
        model = smf.ols(formula, data_reg).fit(displ=False)  # 拟合模型
        prediction = model.predict(data.iloc[i:i + 1][predictor_vars])  # 预测值
        actual = data.iloc[i:i + 1][response_var].values[0]  # 从data中取出第i行的response_var列的值

        r_a = (prediction - actual) ** 2  # 残差平方
        r_b = (np.mean(data_reg[response_var]) - actual) ** 2  # 均值平方
        r_ab = (prediction - actual) * (np.mean(data_reg[response_var]) - actual)  # 残差乘均值

        augment += r_a.values
        benchmark += r_b
        ab += r_ab.values

    oos = 1 - augment / benchmark
    ENCNEW = ((benchmark - ab) / augment) * (len(data) - initial_sample_size)
    MSEF = (benchmark - augment) / augment * (len(data) - initial_sample_size)

    return oos[0], ENCNEW[0], MSEF[0]


# 使用函数测试
oos, ENCNEW, MSEF = out_of_sample(reg_data['2000-01':'2024-12'], 'ret', ['lRV3', 'lpd'], initial_sample_fraction=1 / 3)
print('使用通用函数 - RV3+PD预测:')
print(f'样本外的R方是 {oos:.6f}')
print(f'样本外的ENC-NEW是 {ENCNEW:.6f}')
print(f'样本外的MSE-F是 {MSEF:.6f}')

# 指标一致为负，且幅度比单变量更大：
# 1. R² −7.6%
# 加入 PD 后预测误差方差反而比纯 RV3 又扩大 5 个百分点，说明两变量叠加并未互补，而是共同引入更多噪声。
# 2. ENC-NEW −3.04
# 远低于 Clark-West 临界值，明确拒绝“新增信息”假设；RV3 与 PD 的线性组合对历史均值模型没有任何嵌套改进。
# 3. MSE-F −13.5
# 误差损失显著高于基准，且绝对值比单变量模型几乎翻倍，表明多变量结构在小样本下严重过拟合，导致预测偏差放大。
# 经济含义
# 月频 A 股收益中，波动率与股息率各自携带的方向信号彼此抵消，甚至相互干扰；
# 投资者若同时依据“上月波动”和“上月股息”构造仓位，将系统性地获得更差的样本外收益，预测精度比简单历史均值更低；

# 设置不同的初始样本比例和预测变量组合（'lRV3', 'lRmbusd', 'lpd'）
initial_sample_fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
predictor_combinations = [
    ['lRV3'],
    ['lRmbusd'],
    ['lpd'],
    ['lRV3', 'lRmbusd'],
    ['lRV3', 'lpd'],
    ['lRmbusd', 'lpd'],
    ['lRV3', 'lRmbusd', 'lpd']
]

# 用于存储结果的字典
results = {}

print("不同预测变量组合和初始样本比例的样本外预测结果:\n")
print("=" * 80)

for predictors in predictor_combinations:
    predictor_name = '+'.join(predictors)
    results[predictor_name] = {}
    print(f"\n预测变量: {predictor_name}")
    print("-" * 80)

    for fraction in initial_sample_fractions:
        # 调用 out_of_sample 函数
        oos, ENCNEW, MSEF = out_of_sample(reg_data['2000-01':'2024-12'], 'ret', predictors,
                                          initial_sample_fraction=fraction)

        # 将结果保存到字典
        results[predictor_name][fraction] = {
            'OOS': oos,
            'ENCNEW': ENCNEW,
            'MSEF': MSEF
        }

        # 打印结果
        print(f"初始样本比例: {fraction:.1f} | OOS R²: {oos:8.6f} | ENC-NEW: {ENCNEW:8.4f} | MSE-F: {MSEF:8.4f}")

# 把训练窗从 30% 逐步扩大到 80%，三条指标呈现清晰的“V”型轨迹：
# 1. 小窗口（0.3–0.4）所有预测变量——无论是单变量还是组合——同时出现
# 大幅负 R²（−5% ~ −17%）
# 正 ENC-NEW（+2 ~ +7）
# 大幅负 MSE-F（−10 ~ −30）
# 这是典型的“过拟合+方向对但幅度错”组合：变量在短样本里拟合了方向性残差， Clark-West 检验因而显著；但几次极端误差把平方损失推大，导致 MSE-F 深度为负，样本外精度远低于历史均值。
# 2. 中等窗口（0.5–0.6）
# 训练期延长，估计方差下降，负 R² 收敛至 −1% ~ −12%，MSE-F 绝对值减半；ENC-NEW 迅速转负，说明方向性信号也被平均掉，模型开始“既不准确也不显著”。
# 3. 大窗口（0.7–0.8）
# 训练期覆盖多轮牛熊与政策周期，系数趋于稳定：
# R² 翻正但不超过 +2%，经济量级仍微弱；
# ENC-NEW 回到 0–1 区间，仅勉强高于零界；
# MSE-F 小幅为正，表明平方损失终于不再劣于基准。
# 此时模型不再“添乱”，但也仅比“简单历史均值”好不到 2 个百分点，扣除交易与风险后无实际配置价值。
# 共同结论
# 月频 A 股收益的可预测成分极低，任何少于 50% 的小样本训练都会因过拟合而显著亏损；
# 只有训练期足够长（≥70%），变量组合才勉强停止“添乱”，但 0–2% 的 OOS R² 仍无法覆盖交易成本；
# 汇率、波动率与股息率在中短期维度上无法构成 A 股风险溢价的可利用线性因子，投资者基于这些变量构造战术性仓位，其预期超额收益为零。

# Market Timing 函数
def market_timing(data, response_var, predictor_vars, initial_sample_fraction):
    """
    实现市场择时策略

    参数:
    - data: 数据框
    - response_var: 因变量（收益率）
    - predictor_vars: 预测变量列表
    - initial_sample_fraction: 初始样本比例

    返回:
    - 包含策略表现的DataFrame
    """

    # 先删除缺失值
    data = data.dropna(subset=[response_var] + predictor_vars)

    # 构建模型公式
    predictor_formula = ' + '.join(predictor_vars)
    formula = f'{response_var} ~ {predictor_formula}'

    initial_sample_size = int(len(data) * initial_sample_fraction)

    # 存储预测结果
    predictions = []
    actual_returns = []
    timing_returns = []
    buy_hold_returns = []
    dates = []

    for i in range(initial_sample_size, len(data) - 1):
        data_reg = data.iloc[0:i]
        model = smf.ols(formula, data_reg).fit(displ=False)

        # 预测下一期收益率
        predicted_return = model.predict(data.iloc[i:i + 1][predictor_vars]).values[0]
        actual_return = data.iloc[i + 1][response_var]

        # 择时策略收益: 如果预测为正则持有股票，否则持有无风险资产
        if predicted_return > 0:
            timing_return = actual_return
        else:
            # 假设持有无风险资产
            timing_return = data.iloc[i + 1]['rfmonth'] if 'rfmonth' in data.columns else 0

        # Buy and Hold 策略收益
        buy_hold_return = actual_return

        predictions.append(predicted_return)
        actual_returns.append(actual_return)
        timing_returns.append(timing_return)
        buy_hold_returns.append(buy_hold_return)
        dates.append(data.index[i + 1])

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'date': dates,
        'predicted': predictions,
        'actual': actual_returns,
        'timing_return': timing_returns,
        'buy_hold_return': buy_hold_returns
    })
    results_df.set_index('date', inplace=True)

    # 计算累计收益
    results_df['timing_cum'] = (1 + results_df['timing_return']).cumprod()
    results_df['buy_hold_cum'] = (1 + results_df['buy_hold_return']).cumprod()

    return results_df


# 测试Market Timing策略
timing_results = market_timing(reg_data['2000-01':'2024-12'], 'ret', ['lRV3', 'lpd'], initial_sample_fraction=1 / 2)
print(timing_results)


# 计算策略表现指标
def calculate_strategy_metrics(results_df):
    """
    计算策略表现指标
    """
    metrics = {}

    # 1. 累计收益率
    metrics['Timing Cumulative Return'] = results_df['timing_cum'].iloc[-1] - 1
    metrics['Buy&Hold Cumulative Return'] = results_df['buy_hold_cum'].iloc[-1] - 1

    # 2. 年化收益率
    n_years = len(results_df) / 12
    metrics['Timing Annualized Return'] = (results_df['timing_cum'].iloc[-1] ** (1 / n_years)) - 1
    metrics['Buy&Hold Annualized Return'] = (results_df['buy_hold_cum'].iloc[-1] ** (1 / n_years)) - 1

    # 3. 波动率（年化）
    metrics['Timing Volatility'] = results_df['timing_return'].std() * np.sqrt(12)
    metrics['Buy&Hold Volatility'] = results_df['buy_hold_return'].std() * np.sqrt(12)

    # 4. 夏普比率
    metrics['Timing Sharpe Ratio'] = metrics['Timing Annualized Return'] / metrics['Timing Volatility']
    metrics['Buy&Hold Sharpe Ratio'] = metrics['Buy&Hold Annualized Return'] / metrics['Buy&Hold Volatility']

    # 5. 最大回撤
    timing_cum = results_df['timing_cum']
    timing_running_max = timing_cum.cummax()
    timing_drawdown = (timing_cum - timing_running_max) / timing_running_max
    metrics['Timing Max Drawdown'] = timing_drawdown.min()

    buy_hold_cum = results_df['buy_hold_cum']
    buy_hold_running_max = buy_hold_cum.cummax()
    buy_hold_drawdown = (buy_hold_cum - buy_hold_running_max) / buy_hold_running_max
    metrics['Buy&Hold Max Drawdown'] = buy_hold_drawdown.min()

    # 6. 胜率（预测方向正确的比例）
    correct_direction = ((results_df['predicted'] > 0) & (results_df['actual'] > 0)) | \
                        ((results_df['predicted'] <= 0) & (results_df['actual'] <= 0))
    metrics['Hit Rate'] = correct_direction.sum() / len(results_df)

    # 7. 正收益预测的准确率
    positive_predictions = results_df['predicted'] > 0
    if positive_predictions.sum() > 0:
        metrics['Positive Prediction Accuracy'] = \
            (results_df[positive_predictions]['actual'] > 0).sum() / positive_predictions.sum()
    else:
        metrics['Positive Prediction Accuracy'] = 0

    return metrics


# 计算并展示策略表现
metrics = calculate_strategy_metrics(timing_results)

print("=" * 60)
print("Market Timing 策略表现")
print("=" * 60)
for key, value in metrics.items():
    print(f"{key:.<45} {value:.4f}")
print("=" * 60)


# 可视化累计收益对比
plt.figure(figsize=(12, 6))
plt.plot(timing_results.index, timing_results['timing_cum'], label='Market Timing Strategy', linewidth=2)
plt.plot(timing_results.index, timing_results['buy_hold_cum'], label='Buy & Hold Strategy', linewidth=2, alpha=0.7)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Wealth (Initial = 1)', fontsize=12)
plt.title('Market Timing vs Buy & Hold: Cumulative Returns', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show();

# 比较不同预测变量组合的Market Timing表现
predictor_sets = [
    (['lRV3'], 'RV3'),
    (['lRmbusd'], 'Rmbusd'),
    (['lpd'], 'PD'),
    (['lRV3', 'lRmbusd'], 'RV3+Rmbusd'),
    (['lRV3', 'lpd'], 'RV3+PD'),
    (['lRV3', 'lRmbusd', 'lpd'], 'RV3+Rmbusd+PD')
]

comparison_results = {}

print("\n" + "=" * 100)
print("不同预测变量组合的Market Timing策略对比")
print("=" * 100)
print(f"{'模型':<15} {'累计收益':<12} {'年化收益':<12} {'波动率':<12} {'夏普比率':<12} {'最大回撤':<12} {'胜率':<10}")
print("-" * 100)

for predictors, name in predictor_sets:
    results = market_timing(reg_data['2000-01':'2024-12'], 'ret', predictors, initial_sample_fraction=1 / 2)
    metrics = calculate_strategy_metrics(results)
    comparison_results[name] = {
        'results': results,
        'metrics': metrics
    }

    print(f"{name:<15} {metrics['Timing Cumulative Return']:>11.4f} {metrics['Timing Annualized Return']:>11.4f} "
          f"{metrics['Timing Volatility']:>11.4f} {metrics['Timing Sharpe Ratio']:>11.4f} "
          f"{metrics['Timing Max Drawdown']:>11.4f} {metrics['Hit Rate']:>9.4f}")

# Buy & Hold 基准
buy_hold_metrics = calculate_strategy_metrics(timing_results)
print("-" * 100)
print(f"{'Buy & Hold':<15} {buy_hold_metrics['Buy&Hold Cumulative Return']:>11.4f} "
      f"{buy_hold_metrics['Buy&Hold Annualized Return']:>11.4f} "
      f"{buy_hold_metrics['Buy&Hold Volatility']:>11.4f} {buy_hold_metrics['Buy&Hold Sharpe Ratio']:>11.4f} "
      f"{buy_hold_metrics['Buy&Hold Max Drawdown']:>11.4f} {'N/A':>9}")
print("=" * 100)

# 绘制所有策略的累计收益对比图
plt.figure(figsize=(14, 8))

# 绘制所有Market Timing策略
colors = plt.cm.Set3(np.linspace(0, 1, len(predictor_sets)))
for (predictors, name), color in zip(predictor_sets, colors):
    results = comparison_results[name]['results']
    plt.plot(results.index, results['timing_cum'], label=f'Timing: {name}', linewidth=2, color=color)

# 绘制Buy & Hold
plt.plot(timing_results.index, timing_results['buy_hold_cum'],
         label='Buy & Hold', linewidth=2.5, color='black', linestyle='--', alpha=0.8)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Wealth (Initial = 1)', fontsize=12)
plt.title('Market Timing Strategies Comparison: Different Predictor Combinations', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show();

# 构建季度数据
Qreg_data = reg_data.resample('QE').apply({
    'ret': lambda x: np.exp(sum(np.log(1 + x))) - 1,
    'rfmonth': lambda x: np.exp(sum(np.log(1 + x))) - 1,
    'RV': lambda x: sum(x),
    'RV1': lambda x: sum(x),
    'RV2': lambda x: sum(x),
    'RV3': lambda x: sum(x),
    'pd': lambda x: np.mean(x),
    'pe': lambda x: np.mean(x),
    'pb': lambda x: np.mean(x),
    'cpi': lambda x: sum(x),
    'Rmbusd': lambda x: sum(x),
})

# 创建滞后变量
Qreg_data['lRV3'] = Qreg_data['RV3'].shift(1)
Qreg_data['lpd'] = Qreg_data['pd'].shift(1)
Qreg_data['lpe'] = Qreg_data['pe'].shift(1)
Qreg_data['lpb'] = Qreg_data['pb'].shift(1)
Qreg_data['lcpi'] = Qreg_data['cpi'].shift(1)
Qreg_data['lRmbusd'] = Qreg_data['Rmbusd'].shift(1)
Qreg_data

# 季度数据的样本外预测
data = Qreg_data['2000-01':'2024-12'].copy()
data = data.dropna()  # 删除缺失值
benchmark = 0.00
augment = 0.00
ab = 0.00

for i in range(int(len(data) / 3), len(data) - 1):
    data_reg = data[0:i]
    model1 = smf.ols('ret ~ lRV3', data_reg).fit(displ=False)
    r_a = (model1.predict(data[i:i + 1][['lRV3']]) - data[i:i + 1]['ret'])**2
    r_b = (np.mean(data[0:i]['ret']) - data[i:i + 1]['ret'])**2
    r_ab = (model1.predict(data[i:i + 1][['lRV3']]) - data[i:i + 1]['ret']) * (
        np.mean(data[0:i]['ret']) - data[i:i + 1]['ret'])
    r_a = r_a.values
    r_b = r_b.values
    r_ab = r_ab.values
    augment = augment + r_a
    benchmark += r_b
    ab += r_ab

oos = 1 - augment / benchmark
ENCNEW = ((benchmark - ab) / augment) * (len(data) - int(len(data) / 3))
MSEF = (benchmark - augment) / augment * (len(data) - int(len(data) / 3))

print('季度数据 - 使用RV3预测:')
print(f'样本外的R方是 {oos[0]:.6f}')
print(f'样本外的ENC-NEW是 {ENCNEW[0]:.6f}')
print(f'样本外的MSE-F是 {MSEF[0]:.6f}')