# 张: 学习
# 开发时间: 2021/8/22 17:43
# 浮点类型
  # 浮点数由整数部分和小数部分组成
  # 浮点数存储不精确性： 使用浮点数经行计算时，可能会出现小数位不确定的情况
a=3.1415926
print(a,type(a))
n1=1.1
n2=2.2
print(n1+n2)

# 由于结果出现误差，所以我们解决方案为   导入模块decimal

from decimal import Decimal
print(Decimal('1.1')+Decimal('2.2'))