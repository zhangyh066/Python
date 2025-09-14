# 张: 学习
# 开发时间: 2021/8/30 20:55
'''获取字典中的元素'''
scores={'张三':100,'李四':98,'王五':45}
'''第一种方式，使用[]'''
print(scores['张三'])
#print(scores['a三'])  报错 KeyError: 'a三'

'''第二种方式，使用get（）方法'''
print(scores.get('张三'))
print(scores.get('张1'))   #none
print(scores.get('嘛七',99))  #99是在查找“麻七”所对的value不存在时，所提供的默认值