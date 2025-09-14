# 张: 学习
# 开发时间: 2021/11/24 11:15
import turtle as t
import time
t.color("red","yellow")
t.speed(10)
t.begin_fill()
for _ in range(50):
    t.forward(200)
    t.left(170)
t.end_fill()
time.sleep(1)