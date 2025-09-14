# 张: 学习
# 开发时间: 2021/8/26 23:23
'''多分支结构，多选一执行
从键盘录入一个整数 成绩 来判断范围
90-100 A
80-89  B
70-79  C
60-69  B
0-59   E
小于0或大于100为非法数据（不是成绩有效范围）'''
score=int(input('请输入一个成绩：'))
#判断
if 90<score<=100:
    print('A级')
elif 80<score<90:
    print('B级')
elif 70<score<80:
    print('C级')
elif 60<score<70:
    print('D级')
elif 0<score<60:
    print('E级')
else:
    print('对不起，成绩有误，不在成绩有效范围')