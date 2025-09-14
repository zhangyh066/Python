# 张: 学习
# 开发时间: 2021/9/2 9:09
d={'name':'张三','name':'李四'}  #key不允许重复
print(d)

d={'name':'张三','nikename':'张三'}  #value可以重复
print(d)

lst=[10,20,30]
lst.insert(1,100)
print(lst)
#d={lst:100}  unhashable type: 'list'
