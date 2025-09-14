# 张: 学习
# 开发时间: 2021/8/25 10:26
# 比较运算符，说明比较运算符的结果为bool类型 Ture 或 False
a,b=10,20
print('a>b吗？',a>b)  #False
print('a<b吗？',a<b)  #Ture
print('a<=b吗？',a<=b)  #Ture
print('a>=b吗？',a>=b)  #False
print('a==b吗?',a==b)  #false
print('a!=b吗？',a!=b)  #Ture

'''一个=称为赋值运算符，两个=称为比较运算符
  一个变量由三部分组成，标识，类型，值 
  ==比较的是值还是标识呢？比较的是值
  比较对象的标识使用 is'''

a=10
b=10
print(a==b)  #Ture  说明a与b的value相等
print(a is b)  #Ture  说明a与b的id相等
#以下代码没学过
lst1=[11,22,33,44]
lst2=[11,22,33,44]
print(lst1==lst2)  #value
print(lst1 is lst2)  #id id不相等，两组数不能用同一个地址
print(id(lst1))
print(id(lst2))
print(a is not b )  #False a的id与b的id是相等的
print(lst1 is not lst2)  #Ture lst1的id与lst2的id相等
