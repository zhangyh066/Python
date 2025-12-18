import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
from datetime import datetime
import statsmodels.formula.api as smf

from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from pylab import mpl
import platform

# 根据操作系统设置中文字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS使用Arial Unicode MS
    plt.rcParams['axes.unicode_minus'] = False
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux使用文泉驿微米黑
    plt.rcParams['axes.unicode_minus'] = False

# 1. 解决中文乱码（Windows 下 SimHei 一般都有）
plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 2. 让图在弹出窗口里显示（PyCharm 默认 backend）
plt.rcParams['figure.figsize'] = (6, 4)      # 默认图大小
# 如果想默认保存成 svg，可以再加一行
# plt.rcParams['savefig.format'] = 'svg'

# 3. DataFrame 打印设置
pd.set_option('display.max_columns', None)   # 列不截断
# pd.set_option('display.max_rows',    None)   # 行不截断（需要时再开）



from IPython.core.interactiveshell import InteractiveShell # jupyter运行输出的模块
#显示每一个运行结果
InteractiveShell.ast_node_interactivity = 'all'

# 设置pandas显示选项
pd.set_option('display.max_columns', None)

print(f"当前操作系统: {system}")
print(f"字体设置: {plt.rcParams['font.sans-serif']}")

cross = pd.read_csv('datasets/TRD_Mnth202509.csv')
from pandas.tseries.offsets import MonthEnd
# 处理月份格式
cross['month'] = pd.to_datetime(cross['Trdmnt'], format='%Y-%m') + MonthEnd(1)
# 补齐股票代码 如果不满6位 在前面补上0
cross['Stkcd'] = cross['Stkcd'].apply(lambda x: '{:0>6}'.format(x)) # 6位股票代码
# 重命名列以匹配后续代码
cross.rename(columns={'Mretwd': 'Return', 'Msmvosd': 'floatingvalue', 'Msmvttl': 'totalvalue'}, inplace=True)

# 导入无风险利率数据
rf_data = pd.read_csv('datasets/Marketret_mon_stock2024.csv')
# 处理月份格式
rf_data['month'] = pd.to_datetime(rf_data['month'], format='%b %Y') + MonthEnd(1)
# 只保留需要的列
rf_data = rf_data[['month', 'rfmonth']]

# 合并无风险利率到cross
cross = pd.merge(cross, rf_data, on='month', how='left')

# 添加一个序列 用来统计公司上市的第几个月
cross = cross.sort_values(by=['Stkcd', 'month'])
cross['list_month'] = cross.groupby('Stkcd').cumcount() + 1

# 计算超额收益率
cross['ret'] = cross['Return'] - cross['rfmonth']
cross['floatingvalue'] = cross['floatingvalue'] * 1000
cross['totalvalue'] = cross['totalvalue'] * 1000

print(cross)

# 生成下一个月的收益率

# 方法1：补全所有个股的日期序列，再使用shift
# 创建所有日期和股票代码的完整组合
all_months = pd.DataFrame(cross['month'].unique(), columns=['month'])
all_stocks = pd.DataFrame(cross['Stkcd'].unique(), columns=['Stkcd'])
# 创建笛卡尔积 - 所有股票和所有月份的组合
full_index = all_stocks.merge(all_months, how='cross')

# 将原始数据合并到完整索引中
cross_full = full_index.merge(cross, on=['Stkcd', 'month'], how='left')
# 按股票代码和月份排序
cross_full = cross_full.sort_values(['Stkcd', 'month'])
# 使用shift生成下月收益率
cross_full['next_ret'] = cross_full.groupby('Stkcd')['ret'].shift(-1)

# 只保留原始数据存在的行
cross = cross.merge(cross_full[['Stkcd', 'month', 'next_ret']],
                    on=['Stkcd', 'month'], how='right')

# 添加一个变量 用来统计过去一年的交易日数量之和
cross['Cumsum_tradingday'] = cross.groupby('Stkcd')['Ndaytrd'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())
print("方法1完成：补全日期序列后使用shift")

cross = cross[(cross['month'] >= '1995-01-31') & (cross['month'] <= '2024-12-31')]
print(cross)

from pandas.tseries.offsets import MonthEnd # 月末
Market_ret = pd.read_csv('datasets/Marketret_mon_stock2024.csv')
Market_ret['month'] = pd.to_datetime(Market_ret['month'], format='%b %Y') + MonthEnd(0)
Market_ret.set_index('month', inplace=True)
Market_ret.sort_index(inplace=True)
Market_ret = Market_ret.drop(columns=['Unnamed: 0'])
Market_ret.rename(columns={'ret': 'MKT'}, inplace=True)
print(Market_ret)

cross = pd.merge(cross,Market_ret[['MKT']],left_on='month',right_on='month',how='left')
print(cross)

# 导入价值数据
# 重新读取EP数据并转换month列
EP = pd.read_csv('datasets/EP_individual_mon2024.csv')
EP['Stkcd'] = EP['Stkcd'].apply(lambda x: '{:0>6}'.format(x))

# 转换month列为日期格式
# 根据数据规律: 1991.250000 -> 1991年3月, 1991.333333 -> 1991年4月
# 小数部分 = (month - 1) / 12, 所以 month = round(小数部分 * 12) + 1
EP['year'] = EP['month'].astype(int)
EP['month_decimal'] = EP['month'] - EP['year']
EP['month_num'] = (EP['month_decimal'] * 12).round().astype(int) + 1
# 处理边界情况
EP.loc[EP['month_num'] > 12, 'year'] += 1
EP.loc[EP['month_num'] > 12, 'month_num'] -= 12
EP['month'] = pd.to_datetime(EP['year'].astype(str) + '-' + EP['month_num'].astype(str) + '-01')
EP = EP.drop(['year', 'month_decimal', 'month_num'], axis=1)
EP['month'] = EP['month'] + MonthEnd(1)
EP = EP[['Stkcd', 'month', 'ep', 'ep_recent']]
print(EP)

cross = pd.merge(cross,EP[['Stkcd','month','ep','ep_recent']],on=['Stkcd','month'],how='left')
print(cross)

# 删除最小的30%
fenweishu = pd.DataFrame(
    cross.groupby(['month'])['totalvalue'].quantile(0.3))
fenweishu.columns = ['fenweishu_guimo']
print(fenweishu)

# 合并后再次输出
cross_new = pd.merge(cross,fenweishu,on='month',how='left')
cross_new = cross_new[cross_new['totalvalue'] > cross_new['fenweishu_guimo']]
print(cross_new)

cross_new = cross_new[cross_new['Ndaytrd'] >= 12]
# cross_new = cross_new[cross_new['Clsdt'] >= 5]
cross_new = cross_new[cross_new['list_month'] > 6]
cross_new = cross_new[cross_new['Cumsum_tradingday'] >= 120]
cross_new = cross_new[(cross_new['Markettype'] == 1) | (cross_new['Markettype'] == 4) | (cross_new['Markettype'] == 16)] # 上海A 深圳A 创业板
# ep是完整的
cross_new = cross_new.dropna(subset=['ep'])
print(cross_new)

# 规模、价值分组
guimo = cross_new.groupby(['month'])['totalvalue'].quantile(0.5).to_frame()
guimo.columns = ['guimo']
print(guimo)

jiazhi = cross_new.groupby(['month'])['ep'].quantile([0.3,0.7]).to_frame()
jiazhi.reset_index(inplace=True)
jiazhi = jiazhi.pivot_table(index='month',columns='level_1',values='ep')
jiazhi.columns = ['jiazhi_30','jiazhi_70']
print(jiazhi)

# 数据合并
cross_new = pd.merge(cross_new,guimo,on='month',how='left')
cross_new = pd.merge(cross_new,jiazhi,on='month',how='left')
print(cross_new)

# note the totalvale > guimo will be big, else small
# note the ep > jiazhi_70 will be value, else > jiazhi_30 will be median, else growth
cross_new['size'] = np.where(cross_new['totalvalue'] > cross_new['guimo'],'B','S')
cross_new['value'] = np.where(cross_new['ep'] > cross_new['jiazhi_70'],'V',
                              np.where(cross_new['ep'] > cross_new['jiazhi_30'],'M','G'))
cross_new = cross_new.dropna(subset=['next_ret','totalvalue'])
print(cross_new)

# 计算交叉分组里的加权收益率
def weighted_ret(group):
    return np.average(group['next_ret'], weights=group['totalvalue'])


def calc_portfolio_returns(data):
    portfolios = {}
    for size in ['S', 'B']:
        for value in ['V', 'M', 'G']:
            mask = (data['size'] == size) & (data['value'] == value)
            port_name = f'{size}{value}'
            portfolios[port_name] = (data[mask]
                                     .groupby('month')
                                     .apply(weighted_ret, include_groups=False)
                                     .to_frame(name=port_name))
    return pd.concat(portfolios.values(), axis=1)


six_portfolio = calc_portfolio_returns(cross_new)
# index is month and plus one month
six_portfolio.index = six_portfolio.index + MonthEnd(1)
six_portfolio = six_portfolio['2000-01':]
print(six_portfolio)

# Size Factor
def calc_factors(portfolios):
    # SMB因子
    smb = ((portfolios['SV'] + portfolios['SM'] + portfolios['SG']) / 3 -
           (portfolios['BV'] + portfolios['BM'] + portfolios['BG']) / 3)

    # HML因子
    hml = ((portfolios['SV'] + portfolios['BV']) / 2 -
           (portfolios['SG'] + portfolios['BG']) / 2)

    return pd.DataFrame({
        'SMB': smb,
        'HML': hml
    })


factors = calc_factors(six_portfolio)
print(factors)

# 合并
factors = pd.merge(factors, Market_ret[['MKT']], left_index=True, right_index=True, how='left')
factors = factors[['MKT','SMB','HML']]
print(factors)
factors.to_csv('datasets/factors_3f.csv')

# plot SMB
fig = plt.figure(figsize=(12,6))
plt.plot(factors['SMB'], label='SMB')
plt.legend(loc='upper left')
plt.show();

# Value Factor
# plot HML
fig = plt.figure(figsize=(12,6))
plt.plot(factors['HML'], label='HML')
plt.legend(loc='upper left')
plt.show();

# summary of the factor
factors[:'2016-12'].describe()
factors[:'2016-12'].corr()

# regression
model = smf.ols('SMB ~ 1',
                 data=factors['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
print(model.summary())

# regression
model = smf.ols('HML ~ 1',
                 data=factors['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 2})
print(model.summary())

# 输出价格的走势图
fig = plt.figure(figsize=(12,6))

factors['SMBprice'] = (factors['SMB'] + 1).cumprod()
factors['HMLprice'] = (factors['HML'] + 1).cumprod()

plt.plot(factors['SMBprice'],color='red',label='SMB')
plt.plot(factors['HMLprice'],color='blue',label='HML')
plt.legend(loc='upper left')

xticks = factors.index[::12]
xtick_labels = [label.year for label in xticks]
plt.xticks(xticks, xtick_labels, rotation=90)

plt.show();

# 计算累积收益率
factors['cumulative_return'] = (1 + factors['SMB']).cumprod()

# 计算滚动最大值
factors['rolling_max'] = factors['cumulative_return'].cummax()

# 计算回撤
factors['drawdown'] = factors['cumulative_return'] / factors['rolling_max'] - 1

# 计算最大回撤
max_drawdown = factors['drawdown'].min()
print(f"Maximum Drawdown: {max_drawdown}")

# 找出最大回撤的时间
max_drawdown_end = factors['drawdown'].idxmin() # 返回 Series 中最小值的索引
max_drawdown_start = factors.loc[:max_drawdown_end, 'cumulative_return'].idxmax() # 这部分代码选择了从数据开始到 max_drawdown_end 时间点之间的所有累积收益率

print(f"Maximum Drawdown: {max_drawdown}")
print(f"Maximum Drawdown Start Date: {max_drawdown_start}")
print(f"Maximum Drawdown End Date: {max_drawdown_end}")

# 计算累积收益率
factors['cumulative_return'] = (1 + factors['HML']).cumprod()

# 计算滚动最大值
factors['rolling_max'] = factors['cumulative_return'].cummax()

# 计算回撤
factors['drawdown'] = factors['cumulative_return'] / factors['rolling_max'] - 1

# 计算最大回撤
max_drawdown = factors['drawdown'].min()
print(f"Maximum Drawdown: {max_drawdown}")

# 找出最大回撤的时间
max_drawdown_end = factors['drawdown'].idxmin() # 返回 Series 中最小值的索引
max_drawdown_start = factors.loc[:max_drawdown_end, 'cumulative_return'].idxmax() # 这部分代码选择了从数据开始到 max_drawdown_end 时间点之间的所有累积收益率

print(f"Maximum Drawdown: {max_drawdown}")
print(f"Maximum Drawdown Start Date: {max_drawdown_start}")
print(f"Maximum Drawdown End Date: {max_drawdown_end}")

# 解释规模影响
# 构造totalvalue十分组投资策略
def calc_decile_portfolios(data):
    """
    根据totalvalue构造十分组投资组合
    """
    portfolios = {}

    # 为每个月的每只股票分配十分组标签
    def assign_decile(group):
        # 计算十分位数
        quantiles = [group['totalvalue'].quantile(i / 10) for i in range(11)]
        # 分配到相应的十分组
        labels = pd.cut(group['totalvalue'], bins=quantiles, labels=False, include_lowest=True, duplicates='drop')
        return labels

    data_copy = data.copy()
    data_copy['decile'] = data_copy.groupby('month', group_keys=False).apply(
        lambda x: assign_decile(x), include_groups=False
    )

    # 计算每个十分组的加权收益率
    for i in range(10):
        decile_name = f'P{i + 1}'
        portfolios[decile_name] = (
            data_copy[data_copy['decile'] == i]
            .groupby('month')
            .apply(lambda x: np.average(x['next_ret'], weights=x['totalvalue']), include_groups=False)
            .to_frame(name=decile_name)
        )

    return pd.concat(portfolios.values(), axis=1)


# 计算十分组投资组合
decile_portfolios = calc_decile_portfolios(cross_new)
# 调整索引：投资组合收益率应该对应下一个月
decile_portfolios.index = decile_portfolios.index + MonthEnd(1)
decile_portfolios = decile_portfolios['2000-01':]

print("十分组投资组合描述统计:")
decile_portfolios.describe()

decile_portfolios['size_portfolio'] = decile_portfolios['P1'] - decile_portfolios['P10']
decile_portfolios = pd.merge(decile_portfolios, factors, left_index=True, right_index=True, how='left')
model_size = smf.ols('size_portfolio ~ 1',
                 data=decile_portfolios['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_size.summary())
model_size_3factors = smf.ols('size_portfolio ~ MKT + SMB + HML',
                 data=decile_portfolios['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_size_3factors.summary())

# 解释价值影响
# 构造ep十分组投资策略
def calc_value_decile_portfolios(data):
    """
    根据ep构造十分组投资组合
    """
    portfolios = {}

    # 为每个月的每只股票分配十分组标签
    def assign_decile(group):
        # 计算十分位数
        quantiles = [group['ep'].quantile(i / 10) for i in range(11)]
        # 分配到相应的十分组
        labels = pd.cut(group['ep'], bins=quantiles, labels=False, include_lowest=True, duplicates='drop')
        return labels

    data_copy = data.copy()
    data_copy['ep_decile'] = data_copy.groupby('month', group_keys=False).apply(
        lambda x: assign_decile(x), include_groups=False
    )

    # 计算每个十分组的加权收益率
    for i in range(10):
        decile_name = f'V{i + 1}'
        portfolios[decile_name] = (
            data_copy[data_copy['ep_decile'] == i]
            .groupby('month')
            .apply(lambda x: np.average(x['next_ret'], weights=x['totalvalue']), include_groups=False)
            .to_frame(name=decile_name)
        )

    return pd.concat(portfolios.values(), axis=1)


# 计算十分组投资组合
value_decile_portfolios = calc_value_decile_portfolios(cross_new)
# 调整索引：投资组合收益率应该对应下一个月
value_decile_portfolios.index = value_decile_portfolios.index + MonthEnd(1)
value_decile_portfolios = value_decile_portfolios['2000-01':]

print("价值因子十分组投资组合描述统计:")
value_decile_portfolios.describe()

value_decile_portfolios['value_portfolio'] = value_decile_portfolios['V10'] - value_decile_portfolios['V1']
value_decile_portfolios = pd.merge(value_decile_portfolios, factors, left_index=True, right_index=True, how='left')
model_value = smf.ols('value_portfolio ~ 1',
                 data=value_decile_portfolios['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_value.summary())
model_value_3factors = smf.ols('value_portfolio ~ MKT + SMB + HML',
                 data=value_decile_portfolios['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_value_3factors.summary())

# 解释反转效应
# 构造过去一个月ret十分组投资策略（Reversal效应）
def calc_reversal_decile_portfolios(data):
    """
    根据过去一个月的ret构造十分组投资组合
    """
    portfolios = {}

    # 为每个月的每只股票分配十分组标签
    def assign_decile(group):
        # 计算十分位数
        quantiles = [group['ret'].quantile(i / 10) for i in range(11)]
        # 分配到相应的十分组
        labels = pd.cut(group['ret'], bins=quantiles, labels=False, include_lowest=True, duplicates='drop')
        return labels

    data_copy = data.copy()
    data_copy['ret_decile'] = data_copy.groupby('month', group_keys=False).apply(
        lambda x: assign_decile(x), include_groups=False
    )

    # 计算每个十分组的加权收益率
    for i in range(10):
        decile_name = f'R{i + 1}'
        portfolios[decile_name] = (
            data_copy[data_copy['ret_decile'] == i]
            .groupby('month')
            .apply(lambda x: np.average(x['next_ret'], weights=x['totalvalue']), include_groups=False)
            .to_frame(name=decile_name)
        )

    return pd.concat(portfolios.values(), axis=1)


# 计算十分组投资组合
reversal_decile_portfolios = calc_reversal_decile_portfolios(cross_new)
# 调整索引：投资组合收益率应该对应下一个月
reversal_decile_portfolios.index = reversal_decile_portfolios.index + MonthEnd(1)
reversal_decile_portfolios = reversal_decile_portfolios['2000-01':]

print("Reversal因子十分组投资组合描述统计:")
reversal_decile_portfolios.describe()

reversal_decile_portfolios['reversal_portfolio'] = reversal_decile_portfolios['R1'] - reversal_decile_portfolios['R10']
reversal_decile_portfolios = pd.merge(reversal_decile_portfolios, factors, left_index=True, right_index=True, how='left')
model_reversal = smf.ols('reversal_portfolio ~ 1',
                 data=reversal_decile_portfolios['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_reversal.summary())
model_reversal_3factors = smf.ols('reversal_portfolio ~ MKT + SMB + HML',
                 data=reversal_decile_portfolios['2000-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_reversal_3factors.summary())
