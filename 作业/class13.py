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
import pyreadr

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

from IPython.core.interactiveshell import InteractiveShell
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

# 删除30%
fenweishu = pd.DataFrame(
    cross.groupby(['month'])['totalvalue'].quantile(0.3))
fenweishu.columns = ['fenweishu_guimo']
fenweishu

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

# 个股日收益数据
ret_day = pd.read_csv('datasets/retday2024.csv')
ret_day['Stkcd'] = ret_day['Stkcd'].apply(lambda x: '{:0>6}'.format(x))
ret_day['Day'] = pd.to_datetime(ret_day['Day'], format='%Y-%m-%d')
ret_day['month'] = ret_day['Day'] + MonthEnd(0)
print(ret_day)

# Fama-French 3 Factors 日数据
fama3 = pyreadr.read_r('datasets/FF3_daily2024.RDS')[None]
fama3['Day'] = pd.to_datetime(fama3['Day'], format='%Y-%m-%d')
print(fama3)

ret_day = ret_day.merge(fama3[['Day','mkt.ff','smb.ff','hml.ff']],on='Day',how='left')
print(ret_day)

# 换手率 12个月的换手率
import time
from tqdm import tqdm

# 计算日换手率
ret_day['daily_turnover'] = ret_day['TradingVolume'] / ret_day['All_shares']

# 获取每个月的最后一个交易日
month_end_dates = ret_day.groupby('month')['Day'].max().reset_index()
month_end_dates.columns = ['month', 'month_end_day']

# 计算每个月末往前250天的日期
month_end_dates['start_date'] = month_end_dates['month_end_day'] - pd.Timedelta(days=250)


# 定义函数计算每个股票每个月的12-month turnover
def calc_12m_turnover(group):
    """
    对于每个股票，计算每个月末往前250天的平均日换手率
    """
    results = []
    stkcd = group['Stkcd'].iloc[0]
    group = group.sort_values('Day')

    for _, row in month_end_dates.iterrows():
        month = row['month']
        end_date = row['month_end_day']
        start_date = row['start_date']

        # 筛选该股票在[start_date, end_date]期间的数据
        mask = (group['Day'] >= start_date) & (group['Day'] <= end_date)
        period_data = group.loc[mask, 'daily_turnover']

        if len(period_data) > 0:
            avg_turnover = period_data.mean()
            results.append({
                'Stkcd': stkcd,
                'month': month,
                'turnover_12m': avg_turnover,
                'obs_count': len(period_data)  # 记录观测值数量
            })

    return pd.DataFrame(results)


# 按股票分组计算
print("开始计算12-month turnover...")
print(f"  往前计算天数: 250 天（自然日）")
print(f"  月份数量: {len(month_end_dates)}")
print("=" * 60)

# 使用tqdm添加进度条
grouped = ret_day.groupby('Stkcd')
results_list = []
for stkcd, group in tqdm(grouped, desc="计算Turnover", total=len(grouped)):
    result = calc_12m_turnover(group)
    results_list.append(result)

turnover_12m = pd.concat(results_list, ignore_index=True)

print("=" * 60)
print(f"计算完成:")
print(f"  总观测数: {len(turnover_12m)}")
print(f"  股票数量: {turnover_12m['Stkcd'].nunique()}")
print(f"  平均观测天数: {turnover_12m['obs_count'].mean():.1f}")
print(turnover_12m)

# One-month abnormal turnover
# 计算每个月的1-month turnover（过去20天）和 abnormal turnover
# abnormal turnover = 过去20天平均换手率 - 过去250天平均换手率

# 计算每个月末往前20天的日期
month_end_dates['start_date_20d'] = month_end_dates['month_end_day'] - pd.Timedelta(days=20)


# 定义函数计算每个股票每个月的turnover
def calc_turnover_all(group):
    """
    对于每个股票，计算每个月末往前20天和250天的平均日换手率
    """
    results = []
    stkcd = group['Stkcd'].iloc[0]
    group = group.sort_values('Day')

    for _, row in month_end_dates.iterrows():
        month = row['month']
        end_date = row['month_end_day']
        start_date_250d = row['start_date']
        start_date_20d = row['start_date_20d']

        # 筛选该股票在过去250天的数据
        mask_250d = (group['Day'] >= start_date_250d) & (group['Day'] <= end_date)
        period_data_250d = group.loc[mask_250d, 'daily_turnover']

        # 筛选该股票在过去20天的数据
        mask_20d = (group['Day'] >= start_date_20d) & (group['Day'] <= end_date)
        period_data_20d = group.loc[mask_20d, 'daily_turnover']

        if len(period_data_250d) > 0 and len(period_data_20d) > 0:
            avg_turnover_250d = period_data_250d.mean()
            avg_turnover_20d = period_data_20d.mean()
            abnormal_turnover = avg_turnover_20d - avg_turnover_250d

            results.append({
                'Stkcd': stkcd,
                'month': month,
                'turnover_12m': avg_turnover_250d,
                'turnover_1m': avg_turnover_20d,
                'abnormal_turnover': abnormal_turnover,
                'obs_count_250d': len(period_data_250d),
                'obs_count_20d': len(period_data_20d)
            })

    return pd.DataFrame(results)


# 按股票分组计算
print("开始计算 Turnover 和 Abnormal Turnover...")
print(f"  12-month turnover: 过去250天（自然日）的平均换手率")
print(f"  1-month turnover: 过去20天（自然日）的平均换手率")
print(f"  Abnormal turnover: 1-month turnover - 12-month turnover")
print(f"  月份数量: {len(month_end_dates)}")
print("=" * 60)

# 使用tqdm添加进度条
grouped = ret_day.groupby('Stkcd')
results_list = []
for stkcd, group in tqdm(grouped, desc="计算Turnover", total=len(grouped)):
    result = calc_turnover_all(group)
    results_list.append(result)

turnover_all = pd.concat(results_list, ignore_index=True)

print("=" * 60)
print(f"计算完成:")
print(f"  总观测数: {len(turnover_all)}")
print(f"  股票数量: {turnover_all['Stkcd'].nunique()}")
print(f"  250天平均观测数: {turnover_all['obs_count_250d'].mean():.1f}")
print(f"  20天平均观测数: {turnover_all['obs_count_20d'].mean():.1f}")
print(turnover_all)

# 相关性
# 导入rolling_betas数据
rolling_betas = pd.read_csv('datasets/rolling_betas.csv')
rolling_betas['Stkcd'] = rolling_betas['Stkcd'].apply(lambda x: '{:0>6}'.format(x))
rolling_betas['month'] = pd.to_datetime(rolling_betas['month']) + MonthEnd(0)
print(rolling_betas)

# 将换手率数据合并到cross_new
cross_turnover = pd.merge(cross_new, turnover_all[['Stkcd', 'month', 'turnover_12m', 'turnover_1m', 'abnormal_turnover']],
                          on=['Stkcd', 'month'], how='left')

# 删除缺失值
cross_turnover = cross_turnover.dropna(subset=['turnover_12m', 'abnormal_turnover', 'next_ret'])
print(f"合并后样本量: {len(cross_turnover)}")
cross_turnover

# 合并beta数据到cross_turnover
cross_turnover_corr = pd.merge(cross_turnover, rolling_betas[['Stkcd', 'month', 'beta']],
                               on=['Stkcd', 'month'], how='left')
# 保留2000年1月及以后的数据
cross_turnover_corr = cross_turnover_corr[cross_turnover_corr['month'] >= '2000-01-31']
# 添加size和value变量用于相关性分析
# Size: 使用totalvalue的对数
cross_turnover_corr['size'] = np.log(cross_turnover_corr['totalvalue'])

# Value: 使用EP (已经在数据中)
cross_turnover_corr['value'] = cross_turnover_corr['ep']

print(f"合并beta后样本量: {len(cross_turnover_corr)}")
print(f"beta非缺失样本量: {cross_turnover_corr['beta'].notna().sum()}")
cross_turnover_corr[['Stkcd', 'month', 'abnormal_turnover', 'beta', 'size', 'value']].head(10)


# 计算每个月所有指标之间的截面相关系数矩阵
# 然后在时序上平均

def calc_monthly_correlation_matrix(df):
    """
    计算每个月所有指标两两之间的截面相关系数
    """
    # 选择要分析的变量
    variables = ['abnormal_turnover', 'turnover_12m', 'beta', 'size', 'value', 'next_ret']
    var_names = ['异常换手率', '12月换手率', 'Beta', 'Size (log)', 'Value (EP)', '未来收益率']

    monthly_corr_matrices = []
    valid_months = []

    # 确保所有变量都存在
    available_vars = [v for v in variables if v in df.columns]
    if len(available_vars) < len(variables):
        missing = set(variables) - set(available_vars)
        print(f"警告: 缺少变量 {missing}")
        return None, None, None

    for month in sorted(df['month'].unique()):
        month_data = df[df['month'] == month][variables].dropna()

        if len(month_data) > 10:  # 至少需要10个观测值
            # 计算该月的相关系数矩阵
            corr_matrix = month_data.corr()
            monthly_corr_matrices.append(corr_matrix)
            valid_months.append(month)

    return monthly_corr_matrices, valid_months, var_names


# 计算每月的相关系数矩阵
monthly_corr_matrices, valid_months, var_names = calc_monthly_correlation_matrix(cross_turnover_corr)

if monthly_corr_matrices and len(monthly_corr_matrices) > 0:
    # 在时序上平均
    # 使用 pd.concat 和 groupby 来计算平均矩阵，避免 sum() 的问题
    avg_corr_matrix = pd.concat(monthly_corr_matrices).groupby(level=0).mean()

    # 确保顺序一致
    variables = ['abnormal_turnover', 'turnover_12m', 'beta', 'size', 'value', 'next_ret']
    avg_corr_matrix = avg_corr_matrix.reindex(index=variables, columns=variables)

    # 更新索引和列名为中文
    avg_corr_matrix.index = var_names
    avg_corr_matrix.columns = var_names

    print("=" * 80)
    print("指标间相关系数矩阵（截面相关系数的时序平均）")
    print("=" * 80)
    print(
        f"\n样本期间: {pd.to_datetime(valid_months[0]).strftime('%Y-%m')} 至 {pd.to_datetime(valid_months[-1]).strftime('%Y-%m')}")
    print(f"有效月份数: {len(valid_months)}")
    print(f"每月平均观测数: {cross_turnover_corr.groupby('month').size().mean():.0f}")
    print("\n时序平均相关系数矩阵:")
    avg_corr_matrix
else:
    print("无法计算相关系数矩阵，请检查数据。")

# 美化相关系数矩阵的输出
# 使用热力图可视化

fig, ax = plt.subplots(figsize=(10, 8))

# 创建热力图
im = ax.imshow(
    avg_corr_matrix,   # 数据：相关系数矩阵（二维数组/DataFrame）
    cmap='RdBu',     # 颜色映射：红=负相关，蓝=正相关；_r 表示反转
    aspect='auto',     # 纵横比：'auto' 自适应；'equal' 可使每个格子为正方形
    vmin=-1,           # 颜色下界：最小相关系数对应颜色条下端
    vmax=1             # 颜色上界：最大相关系数对应颜色条上端
)

# 设置刻度标签
ax.set_xticks(np.arange(len(var_names)))
ax.set_yticks(np.arange(len(var_names)))
ax.set_xticklabels(var_names, rotation=45, ha='right')
ax.set_yticklabels(var_names)

# 在每个单元格中显示相关系数值
for i in range(len(var_names)):
    for j in range(len(var_names)):
        text = ax.text(
            j, i,                          # 位置参数: (x, y) 坐标
            f'{avg_corr_matrix.iloc[i, j]:.3f}',  # 文本内容: 格式化的相关系数值(保留3位小数)
            ha="center",                   # 水平对齐方式: "center"=居中, "left"=左对齐, "right"=右对齐
            va="center",                   # 垂直对齐方式: "center"=居中, "top"=顶部对齐, "bottom"=底部对齐
            color="black" if abs(avg_corr_matrix.iloc[i, j]) < 0.5 else "white",  # 文字颜色: 相关系数绝对值<0.5用黑色,否则用白色(便于在深色背景上阅读)
            fontsize=10,                   # 字体大小: 10号字
            fontweight='bold'              # 字体粗细: 'bold'=粗体, 'normal'=正常, 'light'=细体
        )

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('相关系数', rotation=270, labelpad=20)

ax.set_title('指标间相关系数矩阵热力图\n(截面相关系数的时序平均)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show();

# 计算相关系数的统计显著性
# 对于每对变量，提取其时序相关系数并计算t统计量

print("=" * 80)
print("相关系数的统计显著性检验")
print("=" * 80)
print("\nH0: 时序平均相关系数 = 0\n")

# 提取异常换手率与其他变量的相关系数时序
abn_turn_correlations = pd.DataFrame()
for i, var_name in enumerate(var_names):
    if var_name != '异常换手率':
        # 提取每个月异常换手率与该变量的相关系数
        time_series = [matrix.iloc[0, i] for matrix in monthly_corr_matrices]
        abn_turn_correlations[var_name] = time_series

# 计算t检验
significance_results = []
for col in abn_turn_correlations.columns:
    series = abn_turn_correlations[col]
    mean_corr = series.mean()
    std_corr = series.std()
    n = len(series)
    t_stat = mean_corr / (std_corr / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    significance_results.append({
        '变量': col,
        '平均相关系数': mean_corr,
        '标准差': std_corr,
        't统计量': t_stat,
        'p值': p_value,
        '显著性': '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
    })

significance_df = pd.DataFrame(significance_results)
significance_df

print("\n注: *** p<0.01, ** p<0.05, * p<0.1")

# 异象组合收益率
# 将换手率数据合并到cross_new
cross_turnover = pd.merge(cross_new, turnover_all[['Stkcd', 'month', 'turnover_12m', 'turnover_1m', 'abnormal_turnover']],
                          on=['Stkcd', 'month'], how='left')

# 删除缺失值
cross_turnover = cross_turnover.dropna(subset=['turnover_12m', 'abnormal_turnover', 'next_ret'])
print(f"合并后样本量: {len(cross_turnover)}")
print(cross_turnover)

#　12-month Turnover 10分组
# 基于12-month turnover构造10分组投资组合
# 按月份计算turnover_12m的十分位数
cross_turnover['turnover_12m_group'] = cross_turnover.groupby('month')['turnover_12m'].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
)

# 定义计算市值加权收益率的函数
def weighted_avg(group):
    return np.average(group['next_ret'], weights=group['totalvalue'])

# 计算每个分组每个月的等权平均收益率
portfolio_12m_ew = cross_turnover.groupby(['month', 'turnover_12m_group'])['next_ret'].mean().unstack()
portfolio_12m_ew.columns = [f'G{int(i)}' for i in portfolio_12m_ew.columns]
portfolio_12m_ew['H-L'] = portfolio_12m_ew['G1'] - portfolio_12m_ew['G10']

# 计算每个分组每个月的市值加权收益率（修复 FutureWarning）
portfolio_12m_vw = cross_turnover.groupby(['month', 'turnover_12m_group'], group_keys=False)[['next_ret', 'totalvalue']].apply(weighted_avg).unstack()
portfolio_12m_vw.columns = [f'G{int(i)}' for i in portfolio_12m_vw.columns]
portfolio_12m_vw['H-L'] = portfolio_12m_vw['G1'] - portfolio_12m_vw['G10']

# 使用 Newey-West 调整计算 t 值的函数
def calc_newey_west_tvalue(series, maxlags=6):
    """
    使用 Newey-West 方法计算 t 值
    """
    temp_df = pd.DataFrame({'ret': series})
    temp_df = temp_df.dropna()
    if len(temp_df) > 0:
        model = smf.ols('ret ~ 1', data=temp_df).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
        return model.tvalues['Intercept']
    return np.nan

# 计算等权组合的统计量（使用 Newey-West t 值）
stats_12m_ew = pd.DataFrame({
    '平均收益率': portfolio_12m_ew.mean() * 100,
    '标准差': portfolio_12m_ew.std() * 100,
    't值(NW)': portfolio_12m_ew.apply(calc_newey_west_tvalue),
    '夏普比率': portfolio_12m_ew.mean() / portfolio_12m_ew.std() * np.sqrt(12)
})

# 计算市值加权组合的统计量（使用 Newey-West t 值）
stats_12m_vw = pd.DataFrame({
    '平均收益率': portfolio_12m_vw.mean() * 100,
    '标准差': portfolio_12m_vw.std() * 100,
    't值(NW)': portfolio_12m_vw.apply(calc_newey_west_tvalue),
    '夏普比率': portfolio_12m_vw.mean() / portfolio_12m_vw.std() * np.sqrt(12)
})

print("="*60)
print("12-month Turnover 10分组投资组合")
print("="*60)
print("\n【等权加权】各组合月度平均收益率 (%):\n")
print(stats_12m_ew)
print("\n【市值加权】各组合月度平均收益率 (%):\n")
print(stats_12m_vw)

# 基于abnormal turnover构造10分组投资组合
# 按月份计算abnormal_turnover的十分位数
cross_turnover['abnormal_turnover_group'] = cross_turnover.groupby('month')['abnormal_turnover'].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
)

# 计算每个分组每个月的等权平均收益率
portfolio_abn_ew = cross_turnover.groupby(['month', 'abnormal_turnover_group'])['next_ret'].mean().unstack()
portfolio_abn_ew.columns = [f'G{int(i)}' for i in portfolio_abn_ew.columns]
portfolio_abn_ew['H-L'] = portfolio_abn_ew['G1'] - portfolio_abn_ew['G10']

# 计算每个分组每个月的市值加权收益率（修复 FutureWarning）
portfolio_abn_vw = cross_turnover.groupby(['month', 'abnormal_turnover_group'], group_keys=False)[['next_ret', 'totalvalue']].apply(weighted_avg).unstack()
portfolio_abn_vw.columns = [f'G{int(i)}' for i in portfolio_abn_vw.columns]
portfolio_abn_vw['H-L'] = portfolio_abn_vw['G1'] - portfolio_abn_vw['G10']

# 计算等权组合的统计量（使用 Newey-West t 值）
stats_abn_ew = pd.DataFrame({
    '平均收益率': portfolio_abn_ew.mean() * 100,
    '标准差': portfolio_abn_ew.std() * 100,
    't值(NW)': portfolio_abn_ew.apply(calc_newey_west_tvalue),
    '夏普比率': portfolio_abn_ew.mean() / portfolio_abn_ew.std() * np.sqrt(12)
})

# 计算市值加权组合的统计量（使用 Newey-West t 值）
stats_abn_vw = pd.DataFrame({
    '平均收益率': portfolio_abn_vw.mean() * 100,
    '标准差': portfolio_abn_vw.std() * 100,
    't值(NW)': portfolio_abn_vw.apply(calc_newey_west_tvalue),
    '夏普比率': portfolio_abn_vw.mean() / portfolio_abn_vw.std() * np.sqrt(12)
})

print("="*60)
print("Abnormal Turnover 10分组投资组合")
print("="*60)
print("\n【等权加权】各组合月度平均收益率 (%):\n")
print(stats_abn_ew)
print("\n【市值加权】各组合月度平均收益率 (%):\n")
print(stats_abn_vw)

# 投资组合累计收益率图
# 绘制累计收益率图 (等权和市值加权)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 12-month Turnover 等权累计收益率
cum_ret_12m_ew = (1 + portfolio_12m_ew[['G1', 'G10', 'H-L']]).cumprod()
cum_ret_12m_ew.plot(ax=axes[0, 0], linewidth=1.5)
axes[0, 0].set_title('12-month Turnover 投资组合累计收益 (等权)', fontsize=12)
axes[0, 0].set_xlabel('月份')
axes[0, 0].set_ylabel('累计收益率')
axes[0, 0].legend(['Low (G1)', 'High (G10)', 'H-L'], loc='upper left')
axes[0, 0].grid(True, alpha=0.3)

# 12-month Turnover 市值加权累计收益率
cum_ret_12m_vw = (1 + portfolio_12m_vw[['G1', 'G10', 'H-L']]).cumprod()
cum_ret_12m_vw.plot(ax=axes[0, 1], linewidth=1.5)
axes[0, 1].set_title('12-month Turnover 投资组合累计收益 (市值加权)', fontsize=12)
axes[0, 1].set_xlabel('月份')
axes[0, 1].set_ylabel('累计收益率')
axes[0, 1].legend(['Low (G1)', 'High (G10)', 'H-L'], loc='upper left')
axes[0, 1].grid(True, alpha=0.3)

# Abnormal Turnover 等权累计收益率
cum_ret_abn_ew = (1 + portfolio_abn_ew[['G1', 'G10', 'H-L']]).cumprod()
cum_ret_abn_ew.plot(ax=axes[1, 0], linewidth=1.5)
axes[1, 0].set_title('Abnormal Turnover 投资组合累计收益 (等权)', fontsize=12)
axes[1, 0].set_xlabel('月份')
axes[1, 0].set_ylabel('累计收益率')
axes[1, 0].legend(['Low (G1)', 'High (G10)', 'H-L'], loc='upper left')
axes[1, 0].grid(True, alpha=0.3)

# Abnormal Turnover 市值加权累计收益率
cum_ret_abn_vw = (1 + portfolio_abn_vw[['G1', 'G10', 'H-L']]).cumprod()
cum_ret_abn_vw.plot(ax=axes[1, 1], linewidth=1.5)
axes[1, 1].set_title('Abnormal Turnover 投资组合累计收益 (市值加权)', fontsize=12)
axes[1, 1].set_xlabel('月份')
axes[1, 1].set_ylabel('累计收益率')
axes[1, 1].legend(['Low (G1)', 'High (G10)', 'H-L'], loc='upper left')
axes[1, 1].grid(True, alpha=0.3)



print("\n12-month Turnover 各组合累计收益率 (等权):")
print(cum_ret_12m_ew.iloc[-1])
print("\n12-month Turnover 各组合累计收益率 (市值加权):")
print(cum_ret_12m_vw.iloc[-1])
print("\nAbnormal Turnover 各组合累计收益率 (等权):")
print(cum_ret_abn_ew.iloc[-1])
print("\nAbnormal Turnover 各组合累计收益率 (市值加权):")
print(cum_ret_abn_vw.iloc[-1])

plt.tight_layout();
plt.show();


# Fama-French 三因子回归分析
# 读取Fama-French三因子数据
ff3_monthly = pd.read_csv('datasets/factors_3f.csv')
ff3_monthly['month'] = pd.to_datetime(ff3_monthly['month'])
ff3_monthly.set_index('month', inplace=True)
print("Fama-French三因子数据:")
ff3_monthly.head()

# 将组合收益率与三因子数据合并
# Abnormal Turnover 等权组合
portfolio_abn_ew_ff3 = portfolio_abn_ew.merge(ff3_monthly, left_index=True, right_index=True, how='inner')

# Abnormal Turnover 市值加权组合
portfolio_abn_vw_ff3 = portfolio_abn_vw.merge(ff3_monthly, left_index=True, right_index=True, how='inner')


# 定义三因子回归函数
def ff3_regression(portfolio_returns, ff3_data):
    """
    对投资组合进行Fama-French三因子回归
    返回alpha、beta及t值
    """
    results = []

    for col in ['G1', 'G10', 'H-L']:
        if col in portfolio_returns.columns:
            # 准备回归数据
            reg_data = pd.DataFrame({
                'ret': portfolio_returns[col],
                'MKT': ff3_data['MKT'],
                'SMB': ff3_data['SMB'],
                'HML': ff3_data['HML']
            }).dropna()

            # 运行回归（使用Newey-West标准误）
            model = smf.ols('ret ~ MKT + SMB + HML', data=reg_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

            results.append({
                '组合': col,
                'Alpha': model.params['Intercept'] * 100,
                't(Alpha)': model.tvalues['Intercept'],
                'Beta(MKT)': model.params['MKT'],
                't(MKT)': model.tvalues['MKT'],
                'Beta(SMB)': model.params['SMB'],
                't(SMB)': model.tvalues['SMB'],
                'Beta(HML)': model.params['HML'],
                't(HML)': model.tvalues['HML'],
                'R²': model.rsquared
            })

    return pd.DataFrame(results).set_index('组合')


# 对等权组合进行回归
print("=" * 80)
print("Abnormal Turnover 投资组合 - Fama-French 三因子回归 (等权)")
print("=" * 80)
ff3_reg_abn_ew = ff3_regression(portfolio_abn_ew_ff3, portfolio_abn_ew_ff3)
ff3_reg_abn_ew

# 对市值加权组合进行回归
print("=" * 80)
print("Abnormal Turnover 投资组合 - Fama-French 三因子回归 (市值加权)")
print("=" * 80)
ff3_reg_abn_vw = ff3_regression(portfolio_abn_vw_ff3, portfolio_abn_vw_ff3)
print(ff3_reg_abn_vw)

# 深入分析
# 1. 时间序列稳定性分析
# 将样本分为前后两个时期，检验异常换手率效应的时间稳定性
mid_date = portfolio_abn_ew.index[len(portfolio_abn_ew) // 2]

# 前半期
portfolio_abn_ew_first = portfolio_abn_ew[portfolio_abn_ew.index < mid_date]
portfolio_abn_vw_first = portfolio_abn_vw[portfolio_abn_vw.index < mid_date]

# 后半期
portfolio_abn_ew_second = portfolio_abn_ew[portfolio_abn_ew.index >= mid_date]
portfolio_abn_vw_second = portfolio_abn_vw[portfolio_abn_vw.index >= mid_date]


# 计算统计量
def calc_period_stats(portfolio_ew, portfolio_vw, period_name):
    print(f"\n{'=' * 80}")
    print(f"{period_name} ({portfolio_ew.index[0].strftime('%Y-%m')} 至 {portfolio_ew.index[-1].strftime('%Y-%m')})")
    print(f"{'=' * 80}")

    # 等权统计
    stats_ew = pd.DataFrame({
        '平均收益率': portfolio_ew.mean() * 100,
        '标准差': portfolio_ew.std() * 100,
        't值(NW)': portfolio_ew.apply(calc_newey_west_tvalue),
        '夏普比率': portfolio_ew.mean() / portfolio_ew.std() * np.sqrt(12)
    })

    # 市值加权统计
    stats_vw = pd.DataFrame({
        '平均收益率': portfolio_vw.mean() * 100,
        '标准差': portfolio_vw.std() * 100,
        't值(NW)': portfolio_vw.apply(calc_newey_west_tvalue),
        '夏普比率': portfolio_vw.mean() / portfolio_vw.std() * np.sqrt(12)
    })

    print("\n【等权加权】:")
    print(stats_ew[['平均收益率', 't值(NW)', '夏普比率']])
    print("\n【市值加权】:")
    print(stats_vw[['平均收益率', 't值(NW)', '夏普比率']])

    return stats_ew, stats_vw


# 前半期分析
stats_first_ew, stats_first_vw = calc_period_stats(
    portfolio_abn_ew_first, portfolio_abn_vw_first, "前半期")

# 后半期分析
stats_second_ew, stats_second_vw = calc_period_stats(
    portfolio_abn_ew_second, portfolio_abn_vw_second, "后半期")



# 2.1 异常换手率 × 市值（条件排序）
# 双重排序（条件排序）:先按市值分组,再在每个市值组内按异常换手率分组
# 检验异常换手率效应是否在不同市值组中保持一致

# 先按市值分为3组
cross_turnover['size_group'] = cross_turnover.groupby('month')['totalvalue'].transform(
    lambda x: pd.qcut(x, 3, labels=['Small', 'Medium', 'Large'], duplicates='drop')
)

# 在每个市值组内,按异常换手率分为5组
cross_turnover['abn_turn_group_in_size'] = cross_turnover.groupby(['month', 'size_group'], observed=True)[
    'abnormal_turnover'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
)

# 计算双重排序组合的收益率（等权）
double_sort_ew = cross_turnover.groupby(['month', 'size_group', 'abn_turn_group_in_size'], observed=True)[
    'next_ret'].mean().unstack(level=[1, 2])

# 计算双重排序组合的收益率（市值加权）
double_sort_vw = \
cross_turnover.groupby(['month', 'size_group', 'abn_turn_group_in_size'], observed=True, group_keys=False)[
    ['next_ret', 'totalvalue']].apply(weighted_avg).unstack(level=[1, 2])

# 计算每个市值组内的H-L收益率（等权）
results_double_sort_ew = []
for size in ['Small', 'Medium', 'Large']:
    if (size, 1.0) in double_sort_ew.columns and (size, 5.0) in double_sort_ew.columns:
        hl_ret = double_sort_ew[(size, 1.0)] - double_sort_ew[(size, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_double_sort_ew.append({
            '市值组': size,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

double_sort_summary_ew = pd.DataFrame(results_double_sort_ew)

# 计算每个市值组内的H-L收益率（市值加权）
results_double_sort_vw = []
for size in ['Small', 'Medium', 'Large']:
    if (size, 1.0) in double_sort_vw.columns and (size, 5.0) in double_sort_vw.columns:
        hl_ret = double_sort_vw[(size, 1.0)] - double_sort_vw[(size, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_double_sort_vw.append({
            '市值组': size,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

double_sort_summary_vw = pd.DataFrame(results_double_sort_vw)

print("=" * 80)
print("双重排序（条件排序）：异常换手率 × 市值")
print("=" * 80)
print("\n【等权】各市值组内异常换手率H-L组合表现：")
print(double_sort_summary_ew)
print("\n【市值加权】各市值组内异常换手率H-L组合表现：")
print(double_sort_summary_vw)

# 2.2 异常换手率 × 市值（独立排序
# 双重排序（独立排序）：先按市值分组，再在全样本中独立按异常换手率分组
# 这样可以观察在不同市值组中，相同异常换手率水平的股票表现

# 市值分组（使用已有的size_group）
# 异常换手率独立分组（在全样本中分组，不在市值组内分组）
cross_turnover['abn_turn_group_independent'] = cross_turnover.groupby('month')['abnormal_turnover'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
)

# 计算双重排序组合的收益率（等权）
independent_sort_ew = cross_turnover.groupby(['month', 'size_group', 'abn_turn_group_independent'], observed=True)[
    'next_ret'].mean().unstack(level=[1, 2])

# 计算双重排序组合的收益率（市值加权）
independent_sort_vw = \
cross_turnover.groupby(['month', 'size_group', 'abn_turn_group_independent'], observed=True, group_keys=False)[
    ['next_ret', 'totalvalue']].apply(weighted_avg).unstack(level=[1, 2])

# 计算每个市值组内的H-L收益率（等权）
results_independent_ew = []
for size in ['Small', 'Medium', 'Large']:
    if (size, 1.0) in independent_sort_ew.columns and (size, 5.0) in independent_sort_ew.columns:
        hl_ret = independent_sort_ew[(size, 1.0)] - independent_sort_ew[(size, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_independent_ew.append({
            '市值组': size,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

independent_summary_ew = pd.DataFrame(results_independent_ew)

# 计算每个市值组内的H-L收益率（市值加权）
results_independent_vw = []
for size in ['Small', 'Medium', 'Large']:
    if (size, 1.0) in independent_sort_vw.columns and (size, 5.0) in independent_sort_vw.columns:
        hl_ret = independent_sort_vw[(size, 1.0)] - independent_sort_vw[(size, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_independent_vw.append({
            '市值组': size,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

independent_summary_vw = pd.DataFrame(results_independent_vw)

print("=" * 80)
print("双重排序（独立排序）：异常换手率 × 市值")
print("=" * 80)
print("\n【等权】各市值组内异常换手率H-L组合表现：")
print(independent_summary_ew)
print("\n【市值加权】各市值组内异常换手率H-L组合表现：")
print(independent_summary_vw)

# 2.3 异常换手率 × 账面市值比(EP)（条件排序）
# 双重排序（条件排序）：先按EP分组，再在每个EP组内按异常换手率分组
# 检验异常换手率效应是否在不同价值组中保持一致

# 先按EP分为3组
cross_turnover['ep_group'] = cross_turnover.groupby('month')['ep'].transform(
    lambda x: pd.qcut(x, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
)

# 在每个EP组内，按异常换手率分为5组
cross_turnover['abn_turn_group_in_ep'] = cross_turnover.groupby(['month', 'ep_group'], observed=True)[
    'abnormal_turnover'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
)

# 计算双重排序组合的收益率（等权）
double_sort_ep_ew = cross_turnover.groupby(['month', 'ep_group', 'abn_turn_group_in_ep'], observed=True)[
    'next_ret'].mean().unstack(level=[1, 2])

# 计算双重排序组合的收益率（市值加权）
double_sort_ep_vw = \
cross_turnover.groupby(['month', 'ep_group', 'abn_turn_group_in_ep'], observed=True, group_keys=False)[
    ['next_ret', 'totalvalue']].apply(weighted_avg).unstack(level=[1, 2])

# 计算每个EP组内的H-L收益率（等权）
results_double_sort_ep_ew = []
for ep_cat in ['Low', 'Medium', 'High']:
    if (ep_cat, 1.0) in double_sort_ep_ew.columns and (ep_cat, 5.0) in double_sort_ep_ew.columns:
        hl_ret = double_sort_ep_ew[(ep_cat, 1.0)] - double_sort_ep_ew[(ep_cat, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_double_sort_ep_ew.append({
            'EP组': ep_cat,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

double_sort_ep_summary_ew = pd.DataFrame(results_double_sort_ep_ew)

# 计算每个EP组内的H-L收益率（市值加权）
results_double_sort_ep_vw = []
for ep_cat in ['Low', 'Medium', 'High']:
    if (ep_cat, 1.0) in double_sort_ep_vw.columns and (ep_cat, 5.0) in double_sort_ep_vw.columns:
        hl_ret = double_sort_ep_vw[(ep_cat, 1.0)] - double_sort_ep_vw[(ep_cat, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_double_sort_ep_vw.append({
            'EP组': ep_cat,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

double_sort_ep_summary_vw = pd.DataFrame(results_double_sort_ep_vw)

print("=" * 80)
print("双重排序（条件排序）：异常换手率 × EP")
print("=" * 80)
print("\n【等权】各EP组内异常换手率H-L组合表现：")
print(double_sort_ep_summary_ew)
print("\n【市值加权】各EP组内异常换手率H-L组合表现：")
print(double_sort_ep_summary_vw)

# 2.4 异常换手率 × 账面市值比(EP)（独立排序）
# 双重排序（独立排序）：先按EP分组，再在全样本中独立按异常换手率分组

# EP分组（使用已有的ep_group）
# 异常换手率独立分组（使用已有的abn_turn_group_independent）

# 计算双重排序组合的收益率（等权）
independent_sort_ep_ew = cross_turnover.groupby(['month', 'ep_group', 'abn_turn_group_independent'], observed=True)[
    'next_ret'].mean().unstack(level=[1, 2])

# 计算双重排序组合的收益率（市值加权）
independent_sort_ep_vw = \
cross_turnover.groupby(['month', 'ep_group', 'abn_turn_group_independent'], observed=True, group_keys=False)[
    ['next_ret', 'totalvalue']].apply(weighted_avg).unstack(level=[1, 2])

# 计算每个EP组内的H-L收益率（等权）
results_independent_ep_ew = []
for ep_cat in ['Low', 'Medium', 'High']:
    if (ep_cat, 1.0) in independent_sort_ep_ew.columns and (ep_cat, 5.0) in independent_sort_ep_ew.columns:
        hl_ret = independent_sort_ep_ew[(ep_cat, 1.0)] - independent_sort_ep_ew[(ep_cat, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_independent_ep_ew.append({
            'EP组': ep_cat,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

independent_ep_summary_ew = pd.DataFrame(results_independent_ep_ew)

# 计算每个EP组内的H-L收益率（市值加权）
results_independent_ep_vw = []
for ep_cat in ['Low', 'Medium', 'High']:
    if (ep_cat, 1.0) in independent_sort_ep_vw.columns and (ep_cat, 5.0) in independent_sort_ep_vw.columns:
        hl_ret = independent_sort_ep_vw[(ep_cat, 1.0)] - independent_sort_ep_vw[(ep_cat, 5.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_independent_ep_vw.append({
            'EP组': ep_cat,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

independent_ep_summary_vw = pd.DataFrame(results_independent_ep_vw)

print("=" * 80)
print("双重排序（独立排序）：异常换手率 × EP")
print("=" * 80)
print("\n【等权】各EP组内异常换手率H-L组合表现：")
print(independent_ep_summary_ew)
print("\n【市值加权】各EP组内异常换手率H-L组合表现：")
print(independent_ep_summary_vw)

# 3. 换手率水平与异常换手率的交互作用
# 检验12-month turnover水平是否影响abnormal turnover的预测能力
# 3×3双重排序

# 按12-month turnover分为3组
cross_turnover['turnover_12m_tertile'] = cross_turnover.groupby('month')['turnover_12m'].transform(
    lambda x: pd.qcut(x, 3, labels=['Low', 'Medium', 'High'], duplicates='drop')
)

# 在每个12-month turnover组内，按abnormal turnover分为3组
cross_turnover['abn_turn_tertile_in_12m'] = cross_turnover.groupby(['month', 'turnover_12m_tertile'], observed=True)[
    'abnormal_turnover'].transform(
    lambda x: pd.qcut(x, 3, labels=False, duplicates='drop') + 1
)

# 计算3×3组合的收益率
interaction_ew = cross_turnover.groupby(['month', 'turnover_12m_tertile', 'abn_turn_tertile_in_12m'], observed=True)[
    'next_ret'].mean().unstack(level=[1, 2])

# 计算每个12-month turnover组内的H-L收益率
results_interaction = []
for turn_level in ['Low', 'Medium', 'High']:
    if (turn_level, 1.0) in interaction_ew.columns and (turn_level, 3.0) in interaction_ew.columns:
        hl_ret = interaction_ew[(turn_level, 1.0)] - interaction_ew[(turn_level, 3.0)]
        avg_ret = hl_ret.mean() * 100
        t_val = calc_newey_west_tvalue(hl_ret)
        sharpe = hl_ret.mean() / hl_ret.std() * np.sqrt(12)

        results_interaction.append({
            '12月换手率水平': turn_level,
            'H-L平均收益率(%)': avg_ret,
            't值(NW)': t_val,
            '夏普比率': sharpe
        })

interaction_summary = pd.DataFrame(results_interaction)

print("=" * 80)
print("交互作用分析：12-month Turnover × Abnormal Turnover")
print("=" * 80)
print("\n不同12月换手率水平下，异常换手率H-L组合表现：")
print(interaction_summary)

# 4。回报率分析
# 分解H-L组合收益的来源：多头收益 vs 空头收益
# 计算G1(多头)和G10(空头)各自的贡献

# 等权组合
long_ret_ew = portfolio_abn_ew['G1'].mean() * 100
short_ret_ew = portfolio_abn_ew['G10'].mean() * 100
hl_ret_ew = portfolio_abn_ew['H-L'].mean() * 100

long_t_ew = calc_newey_west_tvalue(portfolio_abn_ew['G1'])
short_t_ew = calc_newey_west_tvalue(portfolio_abn_ew['G10'])

# 市值加权组合
long_ret_vw = portfolio_abn_vw['G1'].mean() * 100
short_ret_vw = portfolio_abn_vw['G10'].mean() * 100
hl_ret_vw = portfolio_abn_vw['H-L'].mean() * 100

long_t_vw = calc_newey_west_tvalue(portfolio_abn_vw['G1'])
short_t_vw = calc_newey_west_tvalue(portfolio_abn_vw['G10'])

# 计算多头和空头对H-L收益的贡献比例
long_contrib_ew = (long_ret_ew / hl_ret_ew) * 100 if hl_ret_ew != 0 else 0
short_contrib_ew = (-short_ret_ew / hl_ret_ew) * 100 if hl_ret_ew != 0 else 0

long_contrib_vw = (long_ret_vw / hl_ret_vw) * 100 if hl_ret_vw != 0 else 0
short_contrib_vw = (-short_ret_vw / hl_ret_vw) * 100 if hl_ret_vw != 0 else 0

decomposition_results = pd.DataFrame({
    '组合类型': ['等权', '市值加权'],
    '多头收益率(%)': [long_ret_ew, long_ret_vw],
    '多头t值': [long_t_ew, long_t_vw],
    '空头收益率(%)': [short_ret_ew, short_ret_vw],
    '空头t值': [short_t_ew, short_t_vw],
    'H-L收益率(%)': [hl_ret_ew, hl_ret_vw],
    '多头贡献(%)': [long_contrib_ew, long_contrib_vw],
    '空头贡献(%)': [short_contrib_ew, short_contrib_vw]
})

print("="*80)
print("H-L组合收益分解")
print("="*80)
print("\n多头(G1)和空头(G10)的收益贡献：")
print(decomposition_results)

# 5. 市场状态分析
# 分析异常换手率效应在不同市场状态下的表现
# 区分牛市(市场收益率>0)和熊市(市场收益率<0)

# 合并市场收益率数据
portfolio_abn_with_mkt = portfolio_abn_ew.merge(Market_ret[['MKT']], left_index=True, right_index=True, how='inner')

# 区分牛市和熊市
portfolio_abn_with_mkt['market_state'] = portfolio_abn_with_mkt['MKT'].apply(
    lambda x: 'Up Market' if x > 0 else 'Down Market'
)

# 计算不同市场状态下的H-L收益
def calc_market_state_stats(df, state_label):
    subset = df[df['market_state'] == state_label]
    if len(subset) > 0:
        hl_ret = subset['H-L']
        return {
            '市场状态': state_label,
            '月数': len(subset),
            'H-L平均收益率(%)': hl_ret.mean() * 100,
            't值(NW)': calc_newey_west_tvalue(hl_ret),
            '夏普比率': hl_ret.mean() / hl_ret.std() * np.sqrt(12) if hl_ret.std() > 0 else np.nan
        }
    return None

# 计算统计量
up_market_stats = calc_market_state_stats(portfolio_abn_with_mkt, 'Up Market')
down_market_stats = calc_market_state_stats(portfolio_abn_with_mkt, 'Down Market')

market_state_results = pd.DataFrame([up_market_stats, down_market_stats])

print("="*80)
print("市场状态分析：异常换手率H-L组合在牛市和熊市的表现")
print("="*80)
print(market_state_results)

# 6. 异常换手率的持续性
# 检验异常换手率的自相关性
# 计算当期异常换手率与滞后期的相关性

# 创建滞后变量
# 注意：不能直接使用shift，因为月份可能不连续
# 使用merge方法，通过日期计算来匹配滞后项
from pandas.tseries.offsets import MonthEnd

cross_turnover_sorted = cross_turnover.sort_values(['Stkcd', 'month']).copy()
temp_df = cross_turnover_sorted[['Stkcd', 'month', 'abnormal_turnover']].copy()

for lag in range(1, 13):
    # 构造滞后数据：将数据的月份向后推lag个月，以便匹配到当期
    # 例如：2020-01的数据，月份+1个月变成2020-02，这样在2020-02时就能匹配到lag1（即2020-01的数据）
    lag_data = temp_df.copy()
    lag_data['month'] = lag_data['month'] + MonthEnd(lag)
    lag_data = lag_data.rename(columns={'abnormal_turnover': f'abnormal_turnover_lag{lag}'})

    # 合并回主数据
    cross_turnover_sorted = pd.merge(cross_turnover_sorted, lag_data, on=['Stkcd', 'month'], how='left')

# 计算相关系数
correlations = []
for lag in range(1, 13):
    # 只计算非空值
    valid_data = cross_turnover_sorted[['abnormal_turnover', f'abnormal_turnover_lag{lag}']].dropna()
    if len(valid_data) > 0:
        corr = valid_data.corr().iloc[0, 1]
    else:
        corr = np.nan

    correlations.append({
        '滞后期': f'{lag}个月',
        '相关系数': corr
    })

persistence_df = pd.DataFrame(correlations)

print("=" * 80)
print("异常换手率的持续性分析")
print("=" * 80)
print("\n异常换手率的自相关系数：")
print(persistence_df)

# 7. 相关表现的指标
#计算各组合的最大回撤、信息比率等风险指标


def calculate_risk_metrics(returns):
    """计算风险指标"""
    # 累计收益
    cum_returns = (1 + returns).cumprod()

    # 最大回撤
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # 信息比率 (假设无风险利率为0)
    info_ratio = returns.mean() / returns.std() * np.sqrt(12)

    # 偏度和峰度
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # 下行标准差
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0

    # Sortino比率
    sortino_ratio = returns.mean() / downside_std * np.sqrt(12) if downside_std > 0 else np.nan

    return {
        '平均收益率(%)': returns.mean() * 100,
        '标准差(%)': returns.std() * 100,
        '最大回撤(%)': max_drawdown * 100,
        '夏普比率': info_ratio,
        'Sortino比率': sortino_ratio,
        '偏度': skewness,
        '峰度': kurtosis
    }


# 计算等权组合的风险指标
risk_metrics_ew = []
for col in ['G1', 'G10', 'H-L']:
    metrics = calculate_risk_metrics(portfolio_abn_ew[col])
    metrics['组合'] = col
    risk_metrics_ew.append(metrics)

risk_metrics_ew_df = pd.DataFrame(risk_metrics_ew).set_index('组合')

# 计算市值加权组合的风险指标
risk_metrics_vw = []
for col in ['G1', 'G10', 'H-L']:
    metrics = calculate_risk_metrics(portfolio_abn_vw[col])
    metrics['组合'] = col
    risk_metrics_vw.append(metrics)

risk_metrics_vw_df = pd.DataFrame(risk_metrics_vw).set_index('组合')

print("=" * 80)
print("风险调整后的表现分析")
print("=" * 80)
print("\n【等权组合】风险指标：")
print(risk_metrics_ew_df)
print("\n【市值加权组合】风险指标：")
print(risk_metrics_vw_df)

# 8. 综合总结表
# 汇总所有深入分析的主要结果

print("="*80)
print("异常换手率异象 - 深入分析总结")
print("="*80)

summary_text = f"""
### 1. 时间序列稳定性
- 前半期H-L收益: {stats_first_ew.loc['H-L', '平均收益率']:.3f}% (t={stats_first_ew.loc['H-L', 't值(NW)']:.2f})
- 后半期H-L收益: {stats_second_ew.loc['H-L', '平均收益率']:.3f}% (t={stats_second_ew.loc['H-L', 't值(NW)']:.2f})

### 2. 双重排序分析
异常换手率效应在不同市值和价值组中的表现：
{double_sort_summary_ew.to_string()}

### 3. 换手率交互作用
12-month turnover对abnormal turnover预测力的影响：
{interaction_summary.to_string()}

### 4. 收益分解
- 多头组合(G1)贡献: {long_contrib_ew:.1f}%
- 空头组合(G10)贡献: {short_contrib_ew:.1f}%

### 5. 市场状态
{market_state_results.to_string()}

### 6. 风险指标
等权H-L组合:
- 夏普比率: {risk_metrics_ew_df.loc['H-L', '夏普比率']:.3f}
- 最大回撤: {risk_metrics_ew_df.loc['H-L', '最大回撤(%)']:.2f}%
- Sortino比率: {risk_metrics_ew_df.loc['H-L', 'Sortino比率']:.3f}

### 7. 异常换手率持续性
{persistence_df.to_string()}
"""

print(summary_text)


double_sort_ew = cross_turnover.groupby(['month', 'size_group', 'abn_turn_group_in_size'], observed=True)['next_ret'].mean().to_frame()
print(double_sort_ew)


double_sort_ew = cross_turnover.groupby(['month', 'size_group', 'abn_turn_group_in_size'], observed=True)['next_ret'].mean().unstack(level=[1, 2])
print(double_sort_ew)