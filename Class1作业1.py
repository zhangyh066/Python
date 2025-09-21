import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)#设置列不限制数量
# pd.set_option('display.max_rows',None) 行的话还是不用了
import chardet#编码

def check_encoding(filename):
    rawdata = open(filename, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    confidence = result['confidence']
    return encoding, confidence

file_path = 'datasets/000001.csv'# 添加文件路径
encoding, confidence = check_encoding(file_path)
print(f"Encoding: {encoding}, Confidence: {confidence}")#信心值＞0.7
# 本单元格是导入数据的代码
data = pd.read_csv('datasets/000001.csv') # 这一句是导入CSV文件的命令
data # 出表格
from IPython.core.interactiveshell import InteractiveShell  # 导入Jupyter交互式Shell的核心模块

InteractiveShell.ast_node_interactivity = 'all'#一个单元格每行表达式都会输出
data = pd.read_csv('datasets/000001.csv') # 这一句是导入CSV文件的命令
data
print(data)
type(data.columns)
print(data.columns.values)
print(data.Day)
print(data['Day'])
print(data[['Day']])
print(type(data['Day']))
print(type(data[['Day']]))
print(data[['Day','Close']])
print(data[4:10])#第4行到第9行，共6行
print(data.iloc[4:10,3:8]) #第4行到第9行，共6行，第3列到第7列
print(data.at[3,'Open']) # 按3行索引，列名为open访问
print(data[data['Day'] == "1990/12/21"].Open)# data[data['Day'] == "1990/12/21"].Open#1995/12/21的行然后open开盘价
data['Day'] = pd.to_datetime(data['Day'],format = '%Y/%m/%d')#%Y年%m月%d日
print(data)
data = data.sort_values(by=['Day'], axis=0, ascending=False)
print(data)
help(data.sort_values)
data = data.sort_values(by=['Day'],ascending=True)
print(data)
data.set_index('Day', inplace = True)
print(data)
print(data['1995-12':'2000-04'])
