# 张: 学习
# 开发时间: 2021/8/28 10:46
'''要求输出1到50之间所有5的倍数
   5的倍数的共同点，和5的余数为0的数
'''
for item in range(1,51):
    if item %5==0:
        print(item)

print('-------使用continue-------')
for item in range (1,51):
    if item%5!=0:
      continue