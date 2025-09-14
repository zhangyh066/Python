# 张: 学习
# 开发时间: 2021/8/29 18:30
lst=['hello','world',98,'hello']
print(lst.index('hello'))  #如果列表中有相同元素，只返回列表中相同元素的第一个元素的索引
#print(lst.index('python'))  #ValueError: 'python' is not in list
#print(lst.index('hello',1,3))  #ValueError: 'hello' is not in list

print(lst.index('hello',1,4))