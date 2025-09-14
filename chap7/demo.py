# 张: 学习
# 开发时间: 2021/9/1 19:01
scores={'张三':100,'李四':98,'王五':45}
#获取所有的key
keys=scores.keys()
print(keys)
print(type(keys))
print(list(keys))  #将所有keys组成的视图转成列表

#获取所有的value
values=scores.values()
print(values)
print(type(values))
print(list(values))

#获取所有的key valus键值对
items=scores.items()
print(items)
print(type(items))  #元组（）转换之后的列表元素由元组组成
print(list(items))