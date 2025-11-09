import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime
import statsmodels.formula.api as smf

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

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

# ---------------- 示例：画张图验证能跑 ----------------
if __name__ == '__main__':
    df = pd.DataFrame({'x': np.arange(10),
                       'y': np.random.randn(10)})
    plt.plot(df['x'], df['y'])
    plt.title('中文标题也没问题')
    plt.show()

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
cross_full['next_ret_method1'] = cross_full.groupby('Stkcd')['ret'].shift(-1)

# 只保留原始数据存在的行
cross = cross.merge(cross_full[['Stkcd', 'month', 'next_ret_method1']],
                    on=['Stkcd', 'month'], how='left')

print("方法1完成：补全日期序列后使用shift")
cross[['Stkcd', 'month', 'ret', 'next_ret_method1']].head(20)

# 方法2：截取个股收益率数据，日期加一个月，重命名后合并回去
# 创建下月收益率数据
next_ret_data = cross[['Stkcd', 'month', 'ret']].copy()
# 月份减少一个月
next_ret_data['month'] = next_ret_data['month'] - MonthEnd(1)
# 重命名收益率列
next_ret_data.rename(columns={'ret': 'next_ret_method2'}, inplace=True)

# 合并回原数据
cross = cross.merge(next_ret_data, on=['Stkcd', 'month'], how='left')

print("方法2完成：日期加一个月后合并")
cross[['Stkcd', 'month', 'ret', 'next_ret_method1', 'next_ret_method2']].head(20)

# 验证两种方法的结果是否一致
comparison = cross[['Stkcd', 'month', 'ret', 'next_ret_method1', 'next_ret_method2']].copy()
comparison['difference'] = comparison['next_ret_method1'] - comparison['next_ret_method2']

print("两种方法的差异统计：")
print(comparison['difference'].describe())
print(f"\n完全一致的记录数: {(comparison['difference'].abs() < 1e-10).sum()}")
print(f"总记录数: {len(comparison)}")

# 使用方法2的结果作为最终的next_ret
cross['next_ret'] = cross['next_ret_method2']
# 删除临时列
cross.drop(['next_ret_method1', 'next_ret_method2'], axis=1, inplace=True)

print(cross)

fenweishu = pd.DataFrame(
    cross[cross['ret'].notna()].groupby(['month'])['ret'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
print(fenweishu)

fenweishu = fenweishu.reset_index()
print(fenweishu)

fenweishu = fenweishu.pivot_table(index='month',columns='level_1',values='ret')
print(fenweishu)

fenweishu.columns = ['one','two','three','four','five','six','seven','eight','nine']
print(fenweishu)

portfolio = pd.merge(cross,fenweishu,on='month',how='left')
portfolio = portfolio.dropna(subset=['ret','next_ret'])
print(portfolio)

portfolio['sort'] = np.where(
    portfolio['ret'] <= portfolio['one'], 'P1',
    np.where(
        portfolio['ret'] <= portfolio['two'], 'P2',
        np.where(
            portfolio['ret'] <= portfolio['three'], 'P3',
            np.where(
                portfolio['ret'] <= portfolio['four'], 'P4',
                np.where(
                    portfolio['ret'] <= portfolio['five'], 'P5',
                    np.where(
                        portfolio['ret'] <= portfolio['six'], 'P6',
                        np.where(
                            portfolio['ret'] <= portfolio['seven'], 'P7',
                            np.where(
                                portfolio['ret'] <= portfolio['eight'], 'P8',
                                np.where(
                                    portfolio['ret'] <= portfolio['nine'],
                                    'P9', 'Pmax')))))))))
print(portfolio)

portfolio_mom =  pd.DataFrame(
    portfolio.groupby(['month','sort']).apply(lambda x: np.average(x['next_ret'],weights = x['floatingvalue']),include_groups=False))
print(portfolio_mom)

portfolio_mom = portfolio_mom.reset_index()
portfolio_mom.columns = ['month', 'sort', 'p']
print(portfolio_mom)

# Month plus one month
portfolio_mom['month'] = portfolio_mom['month'] + MonthEnd(1)
# drop NA
portfolio_mom = portfolio_mom.dropna()
print(portfolio_mom)

portfolio_mom = portfolio_mom.pivot_table(index='month',
                                                    columns='sort',
                                                    values='p')
# long lowest return stocks and short highest return stocks
portfolio_mom['My_portfolio'] = portfolio_mom['P1'] - portfolio_mom['Pmax']
print(portfolio_mom)

portfolio_mom = portfolio_mom['1995-01':'2024-12']
print(portfolio_mom)

model_port = smf.ols('My_portfolio ~ 1',
                 data=portfolio_mom['2000-02':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_port.summary())


from pandas.tseries.offsets import MonthEnd # 月末
Market_ret = pd.read_csv('datasets/Marketret_mon_stock2024.csv')
Market_ret['month'] = pd.to_datetime(Market_ret['month'], format='%b %Y') + MonthEnd(0)
Market_ret.set_index('month', inplace=True)
Market_ret.sort_index(inplace=True)
Market_ret = Market_ret.drop(columns=['Unnamed: 0'])
Market_ret.rename(columns={'ret': 'MKT'}, inplace=True)
print(Market_ret)

MYPOR = portfolio_mom[['P1','Pmax','My_portfolio']]
MYPOR = MYPOR.dropna()
MYPOR = pd.merge(MYPOR,Market_ret[['MKT']],left_index=True,right_index=True,how='left')
MYPOR['month'] = pd.date_range(start = '1995',periods=len(MYPOR),freq = 'ME')
MYPOR.set_index('month', inplace = True)
MYPOR = MYPOR["2000":]
print(MYPOR)

model_port = smf.ols('My_portfolio ~ MKT',
                 data=MYPOR['2000-02':'2024-12']).fit(
                     cov_type='HAC', cov_kwds={'maxlags': 6})
print(model_port.summary())

# 计算投资组合的Sharpe Ratio
sharpe_ratio = MYPOR['My_portfolio'].mean() / MYPOR['My_portfolio'].std() * np.sqrt(12)
print(f"Sharpe Ratio: {sharpe_ratio}")

# 还原投资策略的价格
MYPOR_pic = MYPOR['2000-01':'2024-12'].copy()
MYPOR_pic['price_portfolio'] = (1 + MYPOR_pic['My_portfolio']).cumprod()
MYPOR_pic['price_p1'] = (1 + MYPOR_pic['P1']).cumprod()
MYPOR_pic['price_pmax'] = (1 + MYPOR_pic['Pmax']).cumprod()
MYPOR_pic['price_market'] = (1 + MYPOR_pic['MKT']).cumprod()
print(MYPOR_pic)

# 画图
fig = plt.figure(figsize=(12, 4))
plt.plot(
    'price_portfolio',
    '.-g',
    label='Price of My Portfolio',
    linewidth=1.2,
    data=MYPOR_pic)
plt.title("China's Stock Market")
plt.xlabel('Month')
plt.ylabel('Return')

plt.plot(
    'price_market',
    '.-r',
    label='Price of Market',
    linewidth=1.2,
    data=MYPOR_pic)

plt.plot(
    'price_p1',
    '.-y',
    label='Price of Lowest',
    linewidth=1,
    data=MYPOR_pic)

plt.plot(
    'price_pmax',
    '.-g',
    label='Price of Highest',
    linewidth=1,
    data=MYPOR_pic)

# plot legend
plt.legend(loc='upper left')

# 添加网格线
plt.grid(True)

# 添加横线 1
plt.axhline(y=1, color='gray', linewidth=1)

plt.show();







# 计算累积收益率
MYPOR['cumulative_return'] = (1 + MYPOR['My_portfolio']).cumprod()

# 计算滚动最大值
MYPOR['rolling_max'] = MYPOR['cumulative_return'].cummax()

# 计算回撤
MYPOR['drawdown'] = MYPOR['cumulative_return'] / MYPOR['rolling_max'] - 1

# 计算最大回撤
max_drawdown = MYPOR['drawdown'].min()
print(f"Maximum Drawdown: {max_drawdown}")

# 找出最大回撤的时间
max_drawdown_end = MYPOR['drawdown'].idxmin() # 返回 Series 中最小值的索引
max_drawdown_start = MYPOR.loc[:max_drawdown_end, 'cumulative_return'].idxmax() # 这部分代码选择了从数据开始到 max_drawdown_end 时间点之间的所有累积收益率

print(f"Maximum Drawdown: {max_drawdown}")
print(f"Maximum Drawdown Start Date: {max_drawdown_start}")
print(f"Maximum Drawdown End Date: {max_drawdown_end}")


# 构建完整的策略评估函数
def calculate_strategy_metrics(returns, benchmark_returns=None, rf_rate=0.03 / 12):
    """
    计算投资策略的全面评估指标

    Parameters:
    -----------
    returns : pd.Series
        策略收益率序列（月度）
    benchmark_returns : pd.Series, optional
        基准收益率序列（用于计算Information Ratio）
    rf_rate : float
        月度无风险利率，默认年化3%

    Returns:
    --------
    dict : 包含所有评估指标的字典
    """

    metrics = {}

    # 1. 收益指标
    # 累积收益率
    cumulative_return = (1 + returns).prod() - 1
    metrics['累积收益率 (Cumulative Return)'] = f"{cumulative_return:.2%}"

    # 年化收益率
    n_months = len(returns)
    n_years = n_months / 12
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1
    metrics['年化收益率 (Annualized Return)'] = f"{annualized_return:.2%}"

    # 月平均收益率
    avg_monthly_return = returns.mean()
    metrics['月平均收益率 (Average Monthly Return)'] = f"{avg_monthly_return:.2%}"

    # 2. 风险指标
    # 收益率标准差
    volatility = returns.std()
    metrics['月度波动率 (Monthly Volatility)'] = f"{volatility:.2%}"

    # 年化波动率
    annualized_volatility = volatility * np.sqrt(12)
    metrics['年化波动率 (Annualized Volatility)'] = f"{annualized_volatility:.2%}"

    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()
    metrics['最大回撤 (Maximum Drawdown)'] = f"{max_drawdown:.2%}"

    # 下行风险（只考虑负收益的标准差）
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std()
    metrics['下行标准差 (Downside Deviation)'] = f"{downside_deviation:.2%}"

    # VaR (5% 分位数)
    var_95 = returns.quantile(0.05)
    metrics['VaR (95%)'] = f"{var_95:.2%}"

    # CVaR (条件VaR, 最差5%的平均值)
    cvar_95 = returns[returns <= var_95].mean()
    metrics['CVaR (95%)'] = f"{cvar_95:.2%}"

    # 3. 风险调整收益指标
    # 夏普比率
    excess_returns = returns - rf_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    metrics['夏普比率 (Sharpe Ratio)'] = f"{sharpe_ratio:.4f}"

    # 索提诺比率（使用下行风险）
    if downside_deviation > 0:
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(12)
        metrics['索提诺比率 (Sortino Ratio)'] = f"{sortino_ratio:.4f}"
    else:
        metrics['索提诺比率 (Sortino Ratio)'] = "N/A"

    # 卡玛比率（年化收益率/最大回撤）
    if max_drawdown < 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
        metrics['卡玛比率 (Calmar Ratio)'] = f"{calmar_ratio:.4f}"
    else:
        metrics['卡玛比率 (Calmar Ratio)'] = "N/A"

    # 信息比率（如果提供了基准）
    if benchmark_returns is not None:
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        if tracking_error > 0:
            information_ratio = active_returns.mean() / tracking_error * np.sqrt(12)
            metrics['信息比率 (Information Ratio)'] = f"{information_ratio:.4f}"
            metrics['跟踪误差 (Tracking Error)'] = f"{tracking_error * np.sqrt(12):.2%}"
        else:
            metrics['信息比率 (Information Ratio)'] = "N/A"
            metrics['跟踪误差 (Tracking Error)'] = "N/A"

    # 4. 其他重要指标
    # 胜率
    win_rate = (returns > 0).sum() / len(returns)
    metrics['胜率 (Win Rate)'] = f"{win_rate:.2%}"

    # 盈亏比
    avg_gain = returns[returns > 0].mean()
    avg_loss = abs(returns[returns < 0].mean())
    if avg_loss > 0:
        profit_loss_ratio = avg_gain / avg_loss
        metrics['盈亏比 (Profit/Loss Ratio)'] = f"{profit_loss_ratio:.4f}"
    else:
        metrics['盈亏比 (Profit/Loss Ratio)'] = "N/A"

    # 最长回撤期
    # 找到所有创新高的时间点
    is_new_high = cumulative == running_max
    new_high_dates = cumulative[is_new_high].index

    # 计算每次创新高之间的间隔
    if len(new_high_dates) > 1:
        max_drawdown_duration = 0
        for i in range(len(new_high_dates) - 1):
            duration = (new_high_dates[i + 1] - new_high_dates[i]).days / 30  # 转换为月
            max_drawdown_duration = max(max_drawdown_duration, duration)
        metrics['最长回撤期 (月) (Max Drawdown Duration)'] = f"{max_drawdown_duration:.1f}"
    else:
        metrics['最长回撤期 (月) (Max Drawdown Duration)'] = "N/A"

    # 最大回撤起止时间
    max_dd_end = drawdown.idxmin()
    max_dd_start = cumulative.loc[:max_dd_end].idxmax()
    metrics['最大回撤开始时间'] = max_dd_start.strftime('%Y-%m')
    metrics['最大回撤结束时间'] = max_dd_end.strftime('%Y-%m')

    return metrics


# 使用示例
print("=" * 80)
print("投资组合策略评估指标")
print("=" * 80)
portfolio_metrics = calculate_strategy_metrics(
    MYPOR['My_portfolio'],
    benchmark_returns=MYPOR['MKT'],
    rf_rate=0.03 / 12  # 假设年化无风险利率为3%
)

for key, value in portfolio_metrics.items():
    print(f"{key:.<50} {value:>20}")
print("=" * 80)

# 对比分析：我的策略 vs 市场 vs P1 vs Pmax
print("\n" + "=" * 80)
print("不同投资组合的对比分析")
print("=" * 80)

strategies = {
    '反转策略 (My Portfolio)': MYPOR['My_portfolio'],
    '市场组合 (Market)': MYPOR['MKT'],
    '最低收益组 (P1)': MYPOR['P1'],
    '最高收益组 (Pmax)': MYPOR['Pmax']
}

comparison_results = []

for name, returns in strategies.items():
    print(f"\n【{name}】")
    print("-" * 80)
    metrics = calculate_strategy_metrics(returns, benchmark_returns=MYPOR['MKT'], rf_rate=0.03 / 12)

    # 只显示关键指标
    key_metrics = [
        '累积收益率 (Cumulative Return)',
        '年化收益率 (Annualized Return)',
        '年化波动率 (Annualized Volatility)',
        '最大回撤 (Maximum Drawdown)',
        '夏普比率 (Sharpe Ratio)',
        '索提诺比率 (Sortino Ratio)',
        '卡玛比率 (Calmar Ratio)',
        '胜率 (Win Rate)'
    ]

    for key in key_metrics:
        if key in metrics:
            print(f"{key:.<50} {metrics[key]:>20}")

    # 保存用于后续对比
    comparison_results.append({
        '策略': name,
        '年化收益': metrics['年化收益率 (Annualized Return)'],
        '年化波动': metrics['年化波动率 (Annualized Volatility)'],
        '夏普比率': metrics['夏普比率 (Sharpe Ratio)'],
        '最大回撤': metrics['最大回撤 (Maximum Drawdown)']
    })

print("\n" + "=" * 80)

# 创建对比表格
comparison_df = pd.DataFrame(comparison_results)
print("\n策略对比汇总表:")
print(comparison_df.to_string(index=False))
print("\n")


# 绘制完整的回撤图
cumulative_return = (1 + MYPOR['My_portfolio']).cumprod()
running_max = cumulative_return.cummax()
drawdown = (cumulative_return / running_max - 1) * 100  # 转换为百分比

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# 累积收益率和最大值
axes[0].plot(cumulative_return.index, cumulative_return, 'b-', label='Strategy Net Value', linewidth=2)
axes[0].plot(running_max.index, running_max, 'r--', label='Historical Peak Value', linewidth=1, alpha=0.7)
axes[0].set_title('Strategy Net Value Curve', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cumulative Net Value')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# 回撤曲线
axes[1].fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
axes[1].plot(drawdown.index, drawdown, 'r-', linewidth=1)
axes[1].set_title('Drawdown Curve', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_xlabel('Time')
axes[1].grid(True, alpha=0.3)

# 标注最大回撤点
max_dd_value = drawdown.min()
max_dd_date = drawdown.idxmin()
axes[1].plot(max_dd_date, max_dd_value, 'ro', markersize=10)
axes[1].annotate(f'Max Drawdown: {max_dd_value:.2f}%\n{max_dd_date.strftime("%Y-%m")}',
                xy=(max_dd_date, max_dd_value),
                xytext=(20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.tight_layout();
plt.show();

# 年度表现
annual_returns = MYPOR.groupby(MYPOR.index.year).agg({
    'My_portfolio': lambda x: (1 + x).prod() - 1,
    'MKT': lambda x: (1 + x).prod() - 1,
    'P1': lambda x: (1 + x).prod() - 1,
    'Pmax': lambda x: (1 + x).prod() - 1
})

annual_returns.columns = ['Reversal Strategy', 'Market', 'Lowest Return Group', 'Highest Return Group']
annual_returns = annual_returns * 100  # Convert to percentage

print("Annual Returns (%):")
print(annual_returns.round(2))
print(f"\nAverage Annual Return (%): {annual_returns['Reversal Strategy'].mean():.2f}")
print(f"Annual Return Std Dev (%): {annual_returns['Reversal Strategy'].std():.2f}")
print(f"Positive Return Years: {(annual_returns['Reversal Strategy'] > 0).sum()}/{len(annual_returns)}")

# Plot annual returns comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(annual_returns))
width = 0.2

bars1 = ax.bar(x - 1.5*width, annual_returns['Reversal Strategy'], width, label='Reversal Strategy', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, annual_returns['Market'], width, label='Market', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, annual_returns['Lowest Return Group'], width, label='Lowest Return Group', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, annual_returns['Highest Return Group'], width, label='Highest Return Group', alpha=0.8)

ax.set_xlabel('Year')
ax.set_ylabel('Return (%)')
ax.set_title('Annual Returns Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(annual_returns.index, rotation=45)
ax.legend()
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show();

# 季度表现热力图
MYPOR_copy = MYPOR.copy()
MYPOR_copy['Year'] = MYPOR_copy.index.year
MYPOR_copy['Quarter'] = MYPOR_copy.index.quarter

quarterly_returns = MYPOR_copy.groupby(['Year', 'Quarter'])['My_portfolio'].apply(
    lambda x: (1 + x).prod() - 1
) * 100

# 转换为透视表
quarterly_pivot = quarterly_returns.reset_index().pivot(index='Year', columns='Quarter', values='My_portfolio')
quarterly_pivot.columns = ['Q1', 'Q2', 'Q3', 'Q4']

# 添加年度收益率列
quarterly_pivot['Annual'] = annual_returns['Reversal Strategy']

print("\nQuarterly Returns (%):")
print(quarterly_pivot.round(2))

# 绘制热力图
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(quarterly_pivot.values, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)

# 设置坐标轴
ax.set_xticks(np.arange(len(quarterly_pivot.columns)))
ax.set_yticks(np.arange(len(quarterly_pivot.index)))
ax.set_xticklabels(quarterly_pivot.columns)
ax.set_yticklabels(quarterly_pivot.index)

# 在每个格子中显示数值
for i in range(len(quarterly_pivot.index)):
    for j in range(len(quarterly_pivot.columns)):
        value = quarterly_pivot.values[i, j]
        if not np.isnan(value):
            text = ax.text(j, i, f'{value:.1f}%',
                         ha="center", va="center", color="black", fontsize=9)

ax.set_title('Quarterly Returns Heatmap', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Quarter')
ax.set_ylabel('Year')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Return (%)', rotation=270, labelpad=20)

plt.tight_layout()
plt.show();