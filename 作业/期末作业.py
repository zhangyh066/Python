import numpy as np
import pandas as pd
import datetime
from pandas.tseries.offsets import MonthEnd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import platform
import pyreadr
import pandas_datareader.data as web
from pylab import mpl

# 
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
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# 1. 设置时间范围
start_date = datetime.datetime(1926, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

try:
    # 2. 从 Kenneth French 库下载 "F-F_Research_Data_Factors"
    # 这是最标准的月度数据，包含：Mkt-RF, SMB, HML, RF
    ds_factors = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start_date, end_date)

    # pandas_datareader 返回的是一个字典，key 0 通常是月度数据，key 1 是年度数据
    # 我们取月度数据
    df_monthly = ds_factors[0]

    # 3. 提取无风险利率 (RF)
    # 注意：Kenneth French 的数据单位是"百分比" (Percent)
    # 例如：数据中的 '0.4' 代表 '0.4%' (即 0.004)
    # 这一步非常关键！实证中常因为没除以100导致结果完全错误。
    df_monthly['RF_raw'] = df_monthly['RF']  # 保留原始值用于对比
    df_monthly['RF'] = df_monthly['RF'] / 100

    # index重命名
    df_monthly.index.name = 'DATE'
    df_monthly.index = df_monthly.index.to_timestamp() + MonthEnd(0)
    # 打印结果查看
    print("下载成功！数据前 5 行：")
    print(df_monthly[['Mkt-RF', 'RF']].head())

    print("\n数据描述统计：")
    print(df_monthly['RF'].describe())

    # 4. (可选) 保存为 CSV 供后续使用
    # df_monthly.to_csv('FF3_Factors_with_RF.csv')

except Exception as e:
    print(f"下载出错: {e}")
    print("提示：如果因网络问题无法连接，请尝试使用代理或直接访问 Ken French 网站下载 CSV。")

# 导入CRSP美股市场收益率数据
mkt = pd.read_csv('datasets/US stock market return Monthly 2024.csv')
# 查看数据基本信息
print("数据形状:", mkt.shape)
print("\n数据列名:")
print(mkt.columns.tolist())
print("\n数据类型:")
print(mkt.dtypes)
print("\n前5行数据:")
mkt.head()

# 数据处理
# 1. 转换日期格式
mkt['DATE'] = pd.to_datetime(mkt['DATE'],format='%Y-%m-%d')

# 2. 设置日期为索引
mkt = mkt.set_index('DATE')

# 3. 数据字段说明：
# vwretd: 价值加权收益率 (含股息) - Value-Weighted Return (including dividends)
# vwretx: 价值加权收益率 (不含股息) - Value-Weighted Return (excluding dividends)
# ewretd: 等权重收益率 (含股息) - Equal-Weighted Return (including dividends)
# ewretx: 等权重收益率 (不含股息) - Equal-Weighted Return (excluding dividends)
# sprtrn: S&P 500 总收益率 - S&P 500 Total Return
# spindx: S&P 500 指数水平 - S&P 500 Index Level
# totval: 总市值 (千美元) - Total Market Value (in thousands)
# totcnt: 股票总数 - Total Count of Stocks
# usdval: 美国国内股票市值 - US Domestic Stock Value
# usdcnt: 美国国内股票数量 - US Domestic Stock Count

print("数据时间范围:", mkt.index.min(), "至", mkt.index.max())
print("\n描述统计:")
mkt[['vwretd', 'vwretx', 'ewretd', 'ewretx', 'sprtrn']].describe()

# 计算累计收益率
mkt['vwretd_cum'] = (1 + mkt['vwretd']).cumprod()
mkt['ewretd_cum'] = (1 + mkt['ewretd']).cumprod()
mkt['sprtrn_cum'] = (1 + mkt['sprtrn']).cumprod()

# 绘制累计收益率图
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(mkt.index, mkt['vwretd_cum'], label='CRSP 价值加权 (含股息)', linewidth=1.5)
ax.plot(mkt.index, mkt['ewretd_cum'], label='CRSP 等权重 (含股息)', linewidth=1.5)
ax.plot(mkt.index, mkt['sprtrn_cum'], label='S&P 500', linewidth=1.5)

ax.set_xlabel('日期')
ax.set_ylabel('累计收益率 (对数刻度)')
ax.set_title('美股市场累计收益率 (1926-2024)')
ax.set_yscale('log')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# 打印总收益倍数
print(f"\n投资1美元的最终价值 ({mkt.index.min().year}-{mkt.index.max().year}):")
print(f"CRSP 价值加权: ${mkt['vwretd_cum'].iloc[-1]:,.2f}")
print(f"CRSP 等权重: ${mkt['ewretd_cum'].iloc[-1]:,.2f}")
print(f"S&P 500: ${mkt['sprtrn_cum'].iloc[-1]:,.2f}")

plt.tight_layout()
plt.show();

# 计算年化收益率和波动率
def annualized_stats(returns, periods_per_year=12):
    """计算年化收益率、波动率和夏普比率"""
    mean_ret = returns.mean() * periods_per_year
    std_ret = returns.std() * np.sqrt(periods_per_year)
    sharpe = mean_ret / std_ret
    return mean_ret, std_ret, sharpe

# 汇总统计
stats_summary = pd.DataFrame(index=['vwretd', 'ewretd', 'sprtrn'])
stats_summary.index.name = '收益率指标'

for col in ['vwretd', 'ewretd', 'sprtrn']:
    ann_ret, ann_vol, sharpe = annualized_stats(mkt[col])
    stats_summary.loc[col, '年化收益率'] = f"{ann_ret:.2%}"
    stats_summary.loc[col, '年化波动率'] = f"{ann_vol:.2%}"
    stats_summary.loc[col, '夏普比率'] = f"{sharpe:.3f}"
    stats_summary.loc[col, '最大月度收益'] = f"{mkt[col].max():.2%}"
    stats_summary.loc[col, '最小月度收益'] = f"{mkt[col].min():.2%}"
    stats_summary.loc[col, '偏度'] = f"{mkt[col].skew():.3f}"
    stats_summary.loc[col, '峰度'] = f"{mkt[col].kurtosis():.3f}"

stats_summary.index = ['CRSP价值加权', 'CRSP等权重', 'S&P 500']
print("市场收益率统计摘要:")
stats_summary

# 计算超额收益率 (市场收益率 - 无风险利率)
# 合并无风险利率数据
mkt_with_rf = mkt.copy()

# 将FF因子数据的索引也转为Period
df_monthly_copy = df_monthly.copy()

# 合并
mkt_with_rf = pd.merge(
    mkt_with_rf,
    df_monthly_copy[['RF']],
    left_index=True,
    right_index=True,
    how='left'
)

# 计算超额收益率
mkt_with_rf['ret'] = mkt_with_rf['vwretd'] - mkt_with_rf['RF']
mkt_with_rf['rete'] = mkt_with_rf['ewretd'] - mkt_with_rf['RF']

print("超额收益率统计 (市场收益率 - 无风险利率):")
mkt_with_rf[['ret', 'rete']].describe()

# 绘制收益率分布直方图
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, (col, label) in zip(axes, [('vwretd', 'CRSP价值加权'), ('ewretd', 'CRSP等权重')]):
    ax.hist(mkt[col], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(mkt[col].mean(), color='red', linestyle='--', label=f'均值: {mkt[col].mean():.3%}')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('月度收益率')
    ax.set_ylabel('频数')
    ax.set_title(f'{label}月度收益率分布')
    ax.legend()

plt.tight_layout()
plt.show();

# 计算滚动年化收益率和波动率 (36个月滚动窗口)
window = 36

mkt['vwretd_rolling_ret'] = mkt['vwretd'].rolling(window=window).apply(
    lambda x: (1 + x).prod() ** (12/window) - 1, raw=True
)
mkt['vwretd_rolling_vol'] = mkt['vwretd'].rolling(window=window).std() * np.sqrt(12)

# 绘制滚动统计图
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# 滚动年化收益率
axes[0].plot(mkt.index, mkt['vwretd_rolling_ret'], linewidth=1)
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[0].fill_between(mkt.index, 0, mkt['vwretd_rolling_ret'],
                      where=mkt['vwretd_rolling_ret'] >= 0, alpha=0.3, color='green')
axes[0].fill_between(mkt.index, 0, mkt['vwretd_rolling_ret'],
                      where=mkt['vwretd_rolling_ret'] < 0, alpha=0.3, color='red')
axes[0].set_ylabel('年化收益率')
axes[0].set_title(f'{window}个月滚动年化收益率 (CRSP价值加权)')
axes[0].grid(True, alpha=0.3)

# 滚动年化波动率
axes[1].plot(mkt.index, mkt['vwretd_rolling_vol'], linewidth=1, color='orange')
axes[1].set_ylabel('年化波动率')
axes[1].set_xlabel('日期')
axes[1].set_title(f'{window}个月滚动年化波动率 (CRSP价值加权)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show();


# 按年代统计市场收益率
mkt['Decade'] = (mkt.index.year // 10) * 10

decade_stats = mkt.groupby('Decade').agg({
    'vwretd': [
        ('年化收益率', lambda x: (1 + x).prod() ** (12 / len(x)) - 1),
        ('年化波动率', lambda x: x.std() * np.sqrt(12)),
        ('夏普比率', lambda x: ((1 + x).prod() ** (12 / len(x)) - 1) / (x.std() * np.sqrt(12))),
        ('最大回撤月', 'min')
    ]
})

decade_stats.columns = decade_stats.columns.droplevel(0)
decade_stats['年化收益率'] = decade_stats['年化收益率'].apply(lambda x: f"{x:.2%}")
decade_stats['年化波动率'] = decade_stats['年化波动率'].apply(lambda x: f"{x:.2%}")
decade_stats['夏普比率'] = decade_stats['夏普比率'].apply(lambda x: f"{x:.3f}")
decade_stats['最大回撤月'] = decade_stats['最大回撤月'].apply(lambda x: f"{x:.2%}")

print("CRSP价值加权收益率 - 按年代统计:")
decade_stats


# 月度数据聚合为季度数据（Quarterly Aggregation）
# - 对收益类列（以小数表示）做季度复利：(1+r_M1)*(1+r_M2)*(1+r_M3)-1
# - 对指数/数量/市值等“水平”类列取季度末（last）
# - 超额收益（季度）使用更精确的比值法：(Π(1+R_m)) / (Π(1+R_f)) - 1

import os

# 1) 定义需要处理的列
ret_cols = [c for c in ['vwretd', 'ewretd', 'sprtrn', 'RF'] if c in mkt_with_rf.columns]
level_cols = [c for c in ['spindx', 'totval', 'usdval', 'totcnt', 'usdcnt'] if c in mkt_with_rf.columns]

# 2) 定义复利函数
compound = lambda s: (1.0 + s).prod() - 1.0

# 3) 季度收益（复利）
mkt_q_ret = mkt_with_rf[ret_cols].resample('Q').apply(compound)
# 列名加上 _q 后缀，表示“季度”
mkt_q_ret = mkt_q_ret.rename(columns={c: f"{c}_q" for c in mkt_q_ret.columns})

# 4) 更精确的季度超额收益（vw、ew）：(Π(1+R_m)) / (Π(1+R_f)) - 1
vw_cum = (1.0 + mkt_with_rf['vwretd']).resample('Q').prod() if 'vwretd' in mkt_with_rf.columns else None
ew_cum = (1.0 + mkt_with_rf['ewretd']).resample('Q').prod() if 'ewretd' in mkt_with_rf.columns else None
rf_cum = (1.0 + mkt_with_rf['RF']).resample('Q').prod() if 'RF' in mkt_with_rf.columns else None

mkt_q = mkt_q_ret.copy()
if vw_cum is not None and rf_cum is not None:
    mkt_q['vwretd_excess_q'] = (vw_cum / rf_cum) - 1.0
if ew_cum is not None and rf_cum is not None:
    mkt_q['ewretd_excess_q'] = (ew_cum / rf_cum) - 1.0

# 5) 水平类列：取季度末（last）
if level_cols:
    levels_q = mkt_with_rf[level_cols].resample('Q').last()
    mkt_q = mkt_q.join(levels_q)

# 6) 增加季度标签列，便于导出后阅读
mkt_q['Quarter'] = mkt_q.index.to_period('Q').astype(str)

# 7) 简要查看
print('Quarterly DataFrame shape:', mkt_q.shape)
print('\nColumns:')
print(mkt_q.columns.tolist())
print('\nHead:')
mkt_q.head()

# 8) 导出CSV
# os.makedirs('Output', exist_ok=True)
# output_path = 'Output/CRSP_Market_Returns_with_RF_Quarterly.csv'
# mkt_q.to_csv(output_path)
# print(f"\n已保存季度结果至: {output_path}")

# 从FRED下载预测变量

# INDPRO: 工业产出指数 (用于计算增长率)
# GS10: 10年期国债收益率
# TB3MS: 3个月国债收益率

# 选取工业产出指数作为变量
# 选取原因：
# 1.反映实体经济活跃度、领先或同步于企业盈利
# 2.通胀压力信号、影响美联储决策（是否加息）

start = datetime.datetime(1960, 1, 1)
end = datetime.datetime(2024, 12, 31)

try:
    # 下载工业产出指数数据
    indpro = web.DataReader('INDPRO', 'fred', start, end)
    indpro.columns = ['INDPRO']

    # 下载利率数据
    gs5 = web.DataReader('GS5', 'fred', start, end)
    gs5.columns = ['GS5']

    tb3ms = web.DataReader('TB3MS', 'fred', start, end)
    tb3ms.columns = ['TB3MS']

    print("✅ 数据下载成功！")
    print(f"INDPRO数据范围: {indpro.index.min()} 至 {indpro.index.max()}")  # 修改输出变量名
    print(f"GS5数据范围: {gs5.index.min()} 至 {gs5.index.max()}")
    print(f"TB3MS数据范围: {tb3ms.index.min()} 至 {tb3ms.index.max()}")

except Exception as e:
    print(f"❌ 下载出错: {e}")
    print("提示：请检查网络连接或尝试使用代理。")


# 计算预测变量
# 合并所有预测变量数据
predictors = indpro.copy()
predictors = predictors.join(gs5, how='outer')
predictors = predictors.join(tb3ms, how='outer')

# 将索引转换为月末
predictors.index = predictors.index + MonthEnd(0)

# 计算工业产出增长率 (12个月变化率)
predictors['INFL'] = predictors['INDPRO'].pct_change(12)

# 计算期限利差 (Term Spread)
# 注意：GS5和TB3MS的单位是百分比，需要转换为小数
predictors['TMS'] = (predictors['GS5'] - predictors['TB3MS']) / 100

# 查看数据
print("预测变量数据概览：")
print(predictors[['INDPRO', 'GS5', 'TB3MS', 'INFL', 'TMS']].tail(10))

print("\n\n预测变量描述统计：")
predictors[['INFL', 'TMS']].describe()


# 数据合并与滞后处理

# 准备回归数据
# 1. 从mkt_with_rf中提取超额收益率
reg_data = mkt_with_rf[['ret']].copy()  # ret = vwretd - RF

# 2. 合并预测变量
reg_data = reg_data.join(predictors[['INFL', 'TMS']], how='inner')

# 3. 创建滞后变量 (shift 将预测变量向后移动一期)
# 这样当前行的 INFL_lag 和 TMS_lag 对应的是上一期的值
reg_data['INFL_lag'] = reg_data['INFL'].shift(1)
reg_data['TMS_lag'] = reg_data['TMS'].shift(1)

# 4. 删除缺失值
reg_data = reg_data.dropna()

print(f"回归样本期间: {reg_data.index.min()} 至 {reg_data.index.max()}")
print(f"样本量: {len(reg_data)} 个月")
print("\n数据前5行：")
reg_data.head()

print("\n\n数据后5行：")
reg_data.tail()

# 单变量回归 1: 工业产出增长率
model_infl = smf.ols('ret ~ INFL_lag', data=reg_data).fit()

print("=" * 60)
print("模型 1: 工业产出增长率作为预测变量")
print("=" * 60)
print(f"回归方程: R_t = α + β * INFL_{{t-1}} + ε_t")
print(model_infl.summary())

# 单变量回归 2: 期限利差
model_tms = smf.ols('ret ~ TMS_lag', data=reg_data).fit()

print("=" * 60)
print("模型 2: 期限利差作为预测变量")
print("=" * 60)
print(f"回归方程: R_t = α + β * TMS_{{t-1}} + ε_t")
print(model_tms.summary())

# 多变量回归: 工业产出增长率和期限利差同时纳入回归模型。
model_multi = smf.ols('ret ~ INFL_lag + TMS_lag', data=reg_data).fit()

print("=" * 60)
print("模型 3: 多变量回归 (工业产出增长率 + 期限利差)")
print("=" * 60)
print(f"回归方程: R_t = α + β₁ * INFL_{{t-1}} + β₂ * TMS_{{t-1}} + ε_t")
print(model_multi.summary())

# 汇总三个模型的结果
results_summary = pd.DataFrame({
    '模型': ['工业产出增长率', '期限利差', '多变量'],
    'R²': [model_infl.rsquared, model_tms.rsquared, model_multi.rsquared],
    'Adj R²': [model_infl.rsquared_adj, model_tms.rsquared_adj, model_multi.rsquared_adj],
    'F统计量': [model_infl.fvalue, model_tms.fvalue, model_multi.fvalue],
    'F检验p值': [model_infl.f_pvalue, model_tms.f_pvalue, model_multi.f_pvalue],
    '样本量': [model_infl.nobs, model_tms.nobs, model_multi.nobs]
})

# 格式化输出
results_summary['R²'] = results_summary['R²'].apply(lambda x: f"{x:.4f}")
results_summary['Adj R²'] = results_summary['Adj R²'].apply(lambda x: f"{x:.4f}")
results_summary['F统计量'] = results_summary['F统计量'].apply(lambda x: f"{x:.3f}")
results_summary['F检验p值'] = results_summary['F检验p值'].apply(lambda x: f"{x:.4f}")
results_summary['样本量'] = results_summary['样本量'].astype(int)

print("=" * 60)
print("模型比较汇总表")
print("=" * 60)
results_summary

# 样本外预测 - 扩展窗口法 (Expanding Window)

# 设置初始估计窗口 (前20年 = 240个月)
initial_window = 240

# 存储预测结果
oos_predictions = []

# 扩展窗口滚动预测
for t in range(initial_window, len(reg_data)):
    # 使用前t期数据估计模型
    train_data = reg_data.iloc[:t]

    # 估计多变量模型 (使用工业产出增长率和期限利差)
    model_temp = smf.ols('ret ~ INFL_lag + TMS_lag', data=train_data).fit()

    # 预测第t期收益率
    test_obs = reg_data.iloc[[t]]
    pred = model_temp.predict(test_obs)[0]

    oos_predictions.append({
        'date': reg_data.index[t],
        'actual': reg_data.iloc[t]['ret'],
        'predicted': pred,
        'hist_mean': train_data['ret'].mean()  # 历史均值作为基准
    })

# 转换为DataFrame
oos_df = pd.DataFrame(oos_predictions)
oos_df = oos_df.set_index('date')

print(f"样本外预测期间: {oos_df.index.min()} 至 {oos_df.index.max()}")
print(f"样本外预测样本量: {len(oos_df)} 个月")
oos_df.head(10)

# 计算样本外 R²
# OOS R² = 1 - SSE_model / SSE_benchmark
# 基准模型: 历史均值

# 模型预测误差平方和
sse_model = ((oos_df['actual'] - oos_df['predicted']) ** 2).sum()

# 历史均值预测误差平方和 (基准)
sse_benchmark = ((oos_df['actual'] - oos_df['hist_mean']) ** 2).sum()

# 计算 OOS R²
oos_r2 = 1 - sse_model / sse_benchmark

# 也计算样本内 R²
is_r2 = model_multi.rsquared

print("=" * 60)
print("预测能力比较: 样本内 vs 样本外 (基于工业产出增长率和期限利差)")
print("=" * 60)
print(f"样本内 R² (In-Sample):     {is_r2:.4f} ({is_r2*100:.2f}%)")
print(f"样本外 R² (Out-of-Sample): {oos_r2:.4f} ({oos_r2*100:.2f}%)")
print()

if oos_r2 > 0:
    print("✅ OOS R² > 0: 模型基于工业产出增长率和期限利差具有样本外预测能力")
else:
    print("⚠️ OOS R² < 0: 模型的样本外预测能力弱于简单历史均值")

# 绘制预测结果对比图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1: 预测变量时间序列
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
l1, = ax1.plot(reg_data.index, reg_data['INFL_lag'], 'b-', label='工业产出增长率', linewidth=0.8)
l2, = ax1_twin.plot(reg_data.index, reg_data['TMS_lag'], 'r-', label='期限利差', linewidth=0.8)
ax1.set_xlabel('日期')
ax1.set_ylabel('工业产出增长率', color='blue')
ax1_twin.set_ylabel('期限利差', color='red')
ax1.set_title('预测变量时间序列 (工业产出增长率 + 期限利差)')
ax1.legend(handles=[l1, l2], loc='upper right')
ax1.grid(True, alpha=0.3)

# 图2: 样本内拟合值 vs 实际值
ax2 = axes[0, 1]
fitted_values = model_multi.fittedvalues
ax2.scatter(reg_data['ret'], fitted_values, alpha=0.3, s=10)
ax2.plot([reg_data['ret'].min(), reg_data['ret'].max()],
         [reg_data['ret'].min(), reg_data['ret'].max()],
         'r--', label='45度线')
ax2.set_xlabel('实际收益率')
ax2.set_ylabel('拟合收益率')
ax2.set_title(f'样本内拟合效果 (R² = {is_r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 样本外预测 vs 实际收益率 (时间序列)
ax3 = axes[1, 0]
ax3.plot(oos_df.index, oos_df['actual'], 'b-', label='实际收益率', alpha=0.7, linewidth=0.8)
ax3.plot(oos_df.index, oos_df['predicted'], 'r-', label='预测收益率', alpha=0.7, linewidth=0.8)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('日期')
ax3.set_ylabel('收益率')
ax3.set_title(f'样本外预测 vs 实际收益率 (OOS R² = {oos_r2:.4f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: 累计超额收益率 (投资策略表现)
# 假设根据预测值决定投资：预测>0则做多，否则持有无风险资产
oos_df['strategy_ret'] = np.where(oos_df['predicted'] > 0,
                                   oos_df['actual'],
                                   0)  # 简化：预测>0做多，否则不投资
oos_df['cum_actual'] = (1 + oos_df['actual']).cumprod()
oos_df['cum_strategy'] = (1 + oos_df['strategy_ret']).cumprod()

ax4 = axes[1, 1]
ax4.plot(oos_df.index, oos_df['cum_actual'], 'b-', label='买入持有', linewidth=1.2)
ax4.plot(oos_df.index, oos_df['cum_strategy'], 'r-', label='预测策略', linewidth=1.2)
ax4.set_xlabel('日期')
ax4.set_ylabel('累计收益 (初始=1)')
ax4.set_title('基于预测的投资策略表现 (工业产出增长率 + 期限利差)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.show();

# 计算投资策略表现统计
def calc_strategy_stats(returns, annual_periods=12):
    """计算策略表现统计"""
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + returns).prod() ** (annual_periods / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(annual_periods)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()).min() - 1
    return {
        '累计收益率': total_ret,
        '年化收益率': ann_ret,
        '年化波动率': ann_vol,
        '夏普比率': sharpe,
        '最大回撤': max_dd
    }

# 计算两种策略的统计
buy_hold_stats = calc_strategy_stats(oos_df['actual'])
strategy_stats = calc_strategy_stats(oos_df['strategy_ret'])

# 汇总表格
strategy_comparison = pd.DataFrame({
    '买入持有': buy_hold_stats,
    '预测策略': strategy_stats
}).T

# 格式化输出
for col in ['累计收益率', '年化收益率', '年化波动率', '最大回撤']:
    strategy_comparison[col] = strategy_comparison[col].apply(lambda x: f"{x:.2%}")
strategy_comparison['夏普比率'] = strategy_comparison['夏普比率'].apply(lambda x: f"{x:.3f}")

print("=" * 60)
print(f"投资策略表现比较 (基于工业产出增长率和期限利差) ({oos_df.index.min().year} - {oos_df.index.max().year})")
print("=" * 60)
strategy_comparison

# 结论与分析
# 主要发现
# 工业产出指数（Industrial Production Index）

# 工业产出指数对股票市场超额收益率总体呈现正向变动关系
# INDPRO 上升 → 实体经济活跃 → 企业盈利预期改善 → 股票基本面增强 → 股票超额收益上升
# 温和增长阶段通常伴随低通胀 + 宽松或中性货币政策，有利于风险资产表现

# 期限利差 (Term Spread)

# 期限利差对股票市场超额收益率通常呈正向预测关系
# 经济学解释：较大的期限利差往往预示着未来经济扩张，利好股市

# 预测能力

# 样本内R²通常较小（1-3%），这在收益率预测中是正常的
# 样本外R²是评估模型真实预测能力的更可靠指标
# 即使是很小的正OOS R²，在经济意义上也可能具有重要价值

# 局限性
# 预测收益率本质上非常困难，收益率受到大量不可预测因素的影响
# 模型参数的不稳定性可能影响样本外预测表现
# 未来的研究可以考虑加入更多预测变量或使用机器学习方法
