
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

#设置行不限制数量
#pd.set_option('display.max_rows',None)

#设置列不限制数量
pd.set_option('display.max_columns', None)

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

import statsmodels.api as sm


# 假设我们有一个DataFrame 'data'，其中包含个股和市场的月度收益率
# 'data' 包含四列：'month'（月份），'Stkcd'（股票代码），'stock_return'（个股收益率），'market_return'（市场收益率）

# 定义一个函数来计算滚动回归的贝塔和使用的数据点数量
def rolling_beta_per_stock(data, window_months=60):
    # 计算最小需要的数据点数量（2/3的窗口期）
    min_periods = max(1, int(np.ceil(window_months * 2 / 3)))

    print(f"Beta估计设置:")
    print(f"  窗口期: {window_months} 个月")
    print(f"  最少需要: {min_periods} 个月有数据")
    print("=" * 60)

    betas = []
    months = []
    Stkcds = []
    data_counts = []

    # 按股票分组
    grouped = data.groupby('Stkcd')

    for Stkcd, group in grouped:
        group = group.set_index('month').sort_index()
        end_months = group.index.unique()

        for end_month in end_months:
            start_month = end_month - pd.DateOffset(months=window_months)
            window_data = group.loc[start_month:end_month]

            # 只有当数据点数量达到最小要求时才进行回归
            if len(window_data) >= min_periods:
                # 删除缺失值
                window_data_clean = window_data.dropna(subset=['ret', 'MKT'])

                # 再次检查清理后的数据量
                if len(window_data_clean) >= min_periods:
                    X = sm.add_constant(window_data_clean['MKT'])
                    y = window_data_clean['ret']
                    model = sm.OLS(y, X).fit()

                    beta = model.params['MKT']
                    betas.append(beta)
                    months.append(end_month)
                    Stkcds.append(Stkcd)
                    data_counts.append(len(window_data_clean))  # 记录使用的数据点数量

    result_df = pd.DataFrame({'Stkcd': Stkcds, 'month': months, 'beta': betas, 'data_count': data_counts})

    print(f"\n估计完成:")
    print(f"  总观测数: {len(result_df)}")
    print(f"  平均数据点数: {result_df['data_count'].mean():.1f}")
    print(f"  数据点数统计:")
    print(result_df['data_count'].describe())

    return result_df


# 计算每只股票的滚动贝塔和数据点数量
rolling_betas = rolling_beta_per_stock(cross)

# 打印结果
print(rolling_betas)

# save rolling_betas
rolling_betas.to_csv('datasets/rolling_betas.csv', index=False)
print(rolling_betas)

cross_beta = pd.merge(cross,rolling_betas,on=("Stkcd",'month'),how='left')
cross_beta = cross_beta.dropna(subset=['beta'])
print(cross_beta)

cross_beta = cross_beta[cross_beta['Ndaytrd'] >= 7]
cross_beta = cross_beta[cross_beta['Clsdt'] >= 5]
cross_beta = cross_beta[cross_beta['list_month'] > 6]
cross_beta = cross_beta[cross_beta['Cumsum_tradingday'] >= 100]
cross_beta = cross_beta[(cross_beta['Markettype'] == 1) | (cross_beta['Markettype'] == 4) | (cross_beta['Markettype'] == 6)]
print(cross_beta)

fenweishu = pd.DataFrame(
    cross_beta.groupby(['month'])['beta'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
fenweishu = fenweishu.reset_index()
fenweishu = fenweishu.pivot_table(index='month',columns='level_1',values='beta')
fenweishu.columns = ['one','two','three','four','five','six','seven','eight','nine']
print(fenweishu)

portfolio = pd.merge(cross_beta,fenweishu,on='month')
print(portfolio)

portfolio['sort'] = np.where(
    portfolio['beta'] <= portfolio['one'], 'P1',
    np.where(
        portfolio['beta'] <= portfolio['two'], 'P2',
        np.where(
            portfolio['beta'] <= portfolio['three'], 'P3',
            np.where(
                portfolio['beta'] <= portfolio['four'], 'P4',
                np.where(
                    portfolio['beta'] <= portfolio['five'], 'P5',
                    np.where(
                        portfolio['beta'] <= portfolio['six'], 'P6',
                        np.where(
                            portfolio['beta'] <= portfolio['seven'], 'P7',
                            np.where(
                                portfolio['beta'] <= portfolio['eight'], 'P8',
                                np.where(
                                    portfolio['beta'] <= portfolio['nine'],
                                    'P9', 'Pmax')))))))))
portfolio = portfolio.dropna(subset=['floatingvalue','next_ret','beta'])
print(portfolio)

portfolio = portfolio.dropna(subset=['next_ret','floatingvalue','beta'])
portfolio_beta =  pd.DataFrame(
    portfolio.groupby(['month','sort']).apply(lambda x: np.average(x['next_ret'],weights = x['floatingvalue']),include_groups=False))
print(portfolio_beta)

portfolio_beta = portfolio_beta.reset_index()
portfolio_beta.columns = ['month', 'sort', 'p']
portfolio_beta['month'] = portfolio_beta['month'] + MonthEnd(1)
print(portfolio_beta)

portfolio_beta = portfolio_beta.pivot_table(index='month',
                                            columns='sort',
                                            values='p')
portfolio_beta['My_portfolio'] = portfolio_beta['Pmax'] - portfolio_beta['P1']
print(portfolio_beta)

portfolio_beta = portfolio_beta['1995-01':'2024-12']
print(portfolio_beta)

model = smf.ols('My_portfolio ~ 1',
                 data=portfolio_beta['1995-01':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model.summary())

cross_beta2 = cross_beta.copy()
cross_beta2 = cross_beta2.set_index(['Stkcd', 'month']) # 设置multi-index
cross_beta2 = cross_beta2.dropna(subset=['next_ret','beta'])
cross_beta2 = cross_beta2[(cross_beta2.index.get_level_values('month') >= '2000-01-31') &
                          (cross_beta2.index.get_level_values('month') <= '2024-12-31')]
print(cross_beta2)


from linearmodels import FamaMacBeth
model = FamaMacBeth.from_formula('next_ret ~ 1 + beta', data=cross_beta2.dropna(subset=['next_ret','beta']))
## 一般fm回归结果展示的是Newey-West调整后的t值，.fit()中做如下设置
## 其中`bandwidth`是Newey-West滞后阶数，选取方式为lag = 4(T/100) ^ (2/9)
## 若不需要Newey-West调整则去掉括号内所有设置。
# choose bandwidth auto
res = model.fit(cov_type= 'kernel',debiased = False,bandwidth=6)
print(res.summary)

import statsmodels.api as sm

def Fama_MacBeth(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

coef = cross_beta2.dropna(subset=['next_ret','beta']).groupby('month').apply(Fama_MacBeth, 'next_ret', ['beta'])
print(coef)

model_alpha = smf.ols('intercept ~ 1',
                 data=coef['2000-01':'2024-11']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_alpha.summary())

model_beta = smf.ols('beta ~ 1',
                 data=coef['1994-12':'2024-11']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_beta.summary())



