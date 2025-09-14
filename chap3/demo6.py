# 张: 学习
# 开发时间: 2021/8/25 11:15
# 布尔运算符
a,b=1,2
print('----------and 并且-------')
print(a==1 and b==2)  #Ture and Ture-->Ture
print(a==1 and b<2)  #Ture and False-->False
print(a!=1 and b==2)  #False and Ture-->False
print(a!=1 and b!=2)  #False and False-->False


print('--------------or 或者----------')   #两者中只要有一个对的就成立
print(a==1 or b==2)  #Ture and Ture-->Ture
print(a==1 or b<2)  #Ture and False-->Ture
print(a!=1 or b==2)  #False and Ture-->Ture
print(a!=1 or b!=2)  #False and False-->False

print('--------------not 对bool类型的操作数进行取反----------')
f=True
f2=False
print(not f)
print(not f2)

print('--------------in 与 not in有没有在里面----------')
s='halloworld'
print('w'in s)
print('k'in s)
print('w'not in s)
print('k'not in s)