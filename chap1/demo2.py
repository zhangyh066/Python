# 张: 学习
# 开发时间: 2021/8/21 0:00
# 转义字符
print('hello\nworld')  #\  +转义功能的首字母   n--->newland的首字符表示换行
print('hello\tworld')
print('helloooo\tworld')  #\t之后四个字符为一个制表位，若最后一个\t不到四个字符，则会空格
print('hello\rworld')  #\r后面的词把前面的词覆盖掉
print('hello\bworld')  #\b退一个格，把前面的o给推掉了

print('http:\\\\www.baidu.com')  #\\输出成一个\，所以为了能打开网址，\\\\输出成两个\\
print('老师说：\'大家好\'')  #加了斜线之后，'\变成了应该输出的内容，  ' 不是边界


#原字符：不希望字符串中的转义字符不起作用，就使用原字符，就是在字符串之前加上r或R
print(r'hello\nwrold') # 注意事项：原字符串的最后一个字符不能是一个\。如果有就会报错；但可以是两个\\
print(r'hello\nwrold\\')