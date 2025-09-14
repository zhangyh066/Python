# 张: 学习
# 开发时间: 2021/8/26 22:29

money=1000  # 余额
s=int(input('请输入取款金额'))  #取款金额
#判断余额是否充足
if money>=s:
   money=money-s
   print('取款成功，余额为:',money)
else:
   print('余额不足')