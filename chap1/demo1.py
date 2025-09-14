# 张: 学习
# 开发时间: 2021/8/19 16:10

# print的作用
# 可以输出数字
print(520)
print(98.5)

# 可以输出字符串
print('helloworld')

# 含有运算符的表达式并计算表达式结果
print(39556*126523165189612*141984596516515621)


# 将数据输出文件中。 注意点：1.所指定的盘符存在，2.使用file=fp的形式
fp=open('D:/text.txt','a+') #a+的意思：如果文件不存在就创建，文件存在就在文件内容的后面继续追加
print('helloworld',file=fp)
fp.close()

# 不进行换行输出（输出内容在一行当中）
print('hello','world','python')