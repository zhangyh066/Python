# 张: 学习
# 开发时间: 2021/8/30 21:15
# key的判断
scores={'张三':100,'李四':98,'王五':45}
print('张三'in scores)
print('张三'not in scores)

del scores['张三'] #删除指定的键
#scores.clear()  #请空字典的元素
print(scores)

#字典元素的新增
scores['陈六']=98
print(scores)

scores['陈六']=100 #修改元素
print(scores)