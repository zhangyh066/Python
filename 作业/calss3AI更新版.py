# %% 0. 基础 import
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, calendar, os
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

# %% 1. 全局美化
plt.style.use('seaborn-v0_8-paper')
plt.rc.update({'font.family': 'SimHei', 'axes.unicode_minus': False,
               'figure.figsize': (10, 4), 'figure.dpi': 140})
OUT_DIR = Path('images')
OUT_DIR.mkdir(exist_ok=True)

# %% 2. 读数据 → 计算日收益
data = (pd.read_csv('datasets/000001.csv', parse_dates=['Day'], index_col='Day')
          .loc['1995':'2024']                                   # 时段截取
          .assign(Raw_return=lambda x: x.Close / x.Preclose - 1))

# %% 3. 重采样：月/季/年 复利收益
freq_map = {'ME': 'Monthly', 'QE': 'Quarterly', 'YE': 'Annual'}
resampled = {f: (data['Raw_return'].resample(freq).apply(lambda x: (1+x).prod()-1)
                   .to_frame(name='Ret'))
             for freq, f in freq_map.items()}

# %% 4. 月度附加指标
mon = resampled['ME']
mon[['Year', 'Month', 'Month_name']] = (mon.index.year, mon.index.month, mon.index.strftime('%b'))
mon['Rolling_ann_ret'] = (1 + mon.Ret).rolling(12).apply(lambda x: x.prod()-1)
mon['Rolling_ann_vol'] = mon.Ret.rolling(12).std() * np.sqrt(12)

# %% 5. 万能画图函数
def plot_series(df, title, fname, color='#2353E3'):
    fig, ax = plt.subplots()
    ax.plot(df.index, df.Ret, '.-', color=color, label=title.split()[0])
    ax.axhline(0, ls='--', c='gray', lw=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=90)
    ax.set(title=title, xlabel='Date', ylabel='Return')
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / fname, bbox_inches='tight')
    return fig

# 月/季/年 三张线
for freq, name in freq_map.items():
    plot_series(resampled[freq], f"China Stock {name} Return (1995-2024)",
                f"{name}_return.pdf")

# %% 6. 分布 & 热力 & 自相关
sns.histplot(mon.Ret, bins=40, kde=True, color='#2353E3')
plt.title('Monthly Return Distribution'); plt.savefig(OUT_DIR / 'Monthly_dist.pdf'); plt.close()

sns.heatmap(mon.pivot_table(values='Ret', index=mon.index.year, columns=mon.index.month,
                            aggfunc='first') * 100,
            cmap='RdYlGn_r', center=0, fmt='.1f', cbar_kws={'label': '%'})
plt.title('Monthly Return Heatmap'); plt.savefig(OUT_DIR / 'Monthly_heatmap.pdf'); plt.close()

# 自相关（月）
from statsmodels.tsa.stattools import acf
lags = 20
plt.bar(range(1, lags+1), acf(mon.Ret, nlags=lags)[1:])
plt.axhline(0, c='k'); plt.title('Monthly ACF'); plt.savefig(OUT_DIR / 'Monthly_ACF.pdf'); plt.close()

print('✅ 全部完成，图片已保存到', OUT_DIR.resolve())