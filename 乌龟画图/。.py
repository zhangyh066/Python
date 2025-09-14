import tushare  as ts # 引入股票基本数据相关库
import numpy as np
import pandas as pd
import talib # 引入股票衍生变量数据相关库
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score # 引入预测准确度评分函数
from sklearn.model_selection import GridSearchCV # 网格搜索参数调优函数

# #获取数据
# 1. 获取股票基本数据
df = ts.get_k_data('000002', start='2016-01-01',end='2020-12-31')
df = df.set_index('date') # 将日期作为索引值

# 2.简单衍生变量数据构造
df['close-open'] = (df['close']-df['open'])/df['open']
df['high-low'] = (df['high']-df['low'])/df['low']
df['pre_close'] = df['close'].shift(1)
df['price_change'] = df['close']-df['pre_close']
df['p_change'] = (df['close']-df['pre_close'])/df['pre_close']*100

# 3.移动平均线相关数据构造
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()
df.dropna(inplace=True)

# 4. 通过TA_Lib 库构造衍生变量数据
df['RSI'] = talib.RSI(df['close'], timeperiod=12)
df['MOM'] = talib.MOM(df['close'],timeperiod=5)
df['EMA12'] = talib.EMA(df['close'],timeperiod=12)
df['EMA26'] = talib.EMA(df['close'], timeperiod=26)
df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df.dropna(inplace=True)

# # 提取特征变量和目标变量
X = df[
    ['close', 'volume', 'close-open', 'MA5', 'MA10', 'high-low',
     'RSI', 'MOM', 'EMA12', 'MACD', 'MACDsignal', 'MACDhist']
]

"""使用了NumPy库中的where()函数，传入的3个参数的含义分别为判断条件、满足条件的赋值、不满足条件的赋值。其中df['price_change'].shift（-1）
是利用shift()函数将price_change（股价变化）这一列的所有数据向上移动1行，这样就获得了每一行对应的下一天的股价变化。
因此，这里的判断条件就是下一天的股价变化是否大于0，如果大于0，说明下一天股价涨了，则y赋值为1；如果不大于0，说明下一天股价不变或跌了，则y赋值为-1。
预测结果就只有1或-1两种分类。"""
y = np.where(df['price_change'].shift(-1) > 0, 1, -1)

# #划分训练集和测试集
"""需要注意的是，划分要按照时间序列进行，而不能用train_test_split()函数进行随机划分。
这是因为股价的变化趋势具有时间性特征，而随机划分会破坏这种特征，所以需要根据当天的股价数据预测下一天的股价涨跌情况，
而不能根据任意一天的股价数据预测下一天的股价涨跌情况。"""
X_length = X.shape[0] # shape属性获取X的行数和列数，shape[0]即为行数
split = int(X_length*0.9)
X_train, X_test = X[:split], X[:split]
y_train, y_test = y[:split], y[:split]

# # 模型搭建
"""决策树的最大深度max_depth设置为3，即每个决策树最多只有3层；弱学习器（即决策树模型）的个数n_estimators设置为10，
即该随机森林中共有10个决策树；
叶子节点的最小样本数min_samples_leaf设置为10，即如果叶子节点的样本数小于10则停止分裂；
随机状态参数random_state的作用是使每次运行结果保持一致，这里设置的数字1没有特殊含义，可以换成其他数字。"""
rfc = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_leaf=10, random_state=1)

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

print('预测情况:\n', y_pred)

"""预测属于各个分类的概率，获得的y_pred_proba是一个二维数组，其第1列为分类为-1（下一天股价不变或下跌）的概率，
第2列为分类为1（下一天股价上涨）的概率"""
y_pred_proba = rfc.predict_proba(X_test)

# 模型准确度评估
score = accuracy_score(y_pred, y_test)
print(score)
print(rfc.score(X_test,y_test)) # 和score值一样，等价操作

# # 分析特征变量的重要性
print(rfc.feature_importances_)

# 对特征按照计算出来的特征重要性进行排序（降序）
features = X.columns
importances= rfc.feature_importances_
a = pd.DataFrame()
a['特征'] = features
a['特征重要性'] = importances
a = a.sort_values('特征重要性', ascending=False)
print(a)

#  #参数调优
"""n_estimators参数的候选值范围为{5，10，20}，max_depth参数的候选值范围为{2，3，4，5}，
min_samples_leaf参数的候选值范围为{5，10，20，30}。"""
parameters = {'n_estimators': [5, 10, 20], 'max_depth': [2, 3, 4, 5], 'min_samples_leaf': [5, 10, 20, 30]}
new_rfc = RandomForestClassifier(random_state=1)  # 构建的随机森林模型

"""设置cv参数为6，表示交叉验证6次；设置模型评估标准scoring参数为'accuracy'，即以准确度作为评估标准，
如果设置成'roc_auc'则表示以ROC曲线的AUC值作为评估标准。"""
grid_search = GridSearchCV(new_rfc, parameters, cv=6, scoring='accuracy')

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)  # 打印调参最优结果

X_test['prediction'] = rfc.predict(X_test)
# 计算每天的股价变化率，即（当天的收盘价-前一天的收盘价）/前一天的收盘价。
X_test['p_change'] = (X_test['close'] - X_test['close'].shift(1))/X_test['close'].shift(1)

X_test['origin'] = (X_test['p_change']+1).cumprod()
"""因为是根据当天的股价数据预测下一天的股价涨跌情况，如果预测为1，则在下一天买入，如果预测为-1，则在下一天卖出，
所以这里通过shift（1）将预测结果这一列向下移一行，这样才能和下一天的股价变化率相匹配，再用cumprod()函数来计算收益率。"""
X_test['strategy'] = (X_test['prediction'].shift(1)*X_test['p_change']+1).cumprod()


X_test[['strategy', 'origin']].dropna().plot()
plt.gcf().autofmt_xdate()
plt.show()
