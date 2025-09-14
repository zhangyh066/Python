# 张: 学习
# 开发时间: 2021/8/28 10:30
'''从键盘去录入密码，最多录入三次，如果正确就结束循环'''
for item in range(3):
    pwd=input('请输入密码')
    if pwd=='8888':
       print('密码正确')
       break
    else:
        print('密码不正确')
        