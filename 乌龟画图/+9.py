# 张: 学习
# 开发时间: 2021/12/6 22:14

from turtle import *

def drawTriangle(points,colors,t):
   t.fillcolor(colors)
   t.up()
   t.goto(points[0])
   t.down()
   t.begin_fill()
   t.goto(points[1])
   t.goto(points[2])
   t.goto(points[0])
   t.end_fill()

def getMid(p1,p2):
   return ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

def sierpinski(points,degree,myTurtle):
   colormap = ['red','green','blue','white','pink','orange','yellow']
   drawTriangle(points,colormap[degree],myTurtle)
   if degree > 0:
      sierpinski([points[0],getMid(points[0],points[1]),getMid(points[0],points[2])],degree-1,myTurtle)
      sierpinski([points[1],getMid(points[0],points[1]),getMid(points[1],points[2])],degree-1,myTurtle)
      sierpinski([points[2],getMid(points[2],points[1]),getMid(points[0],points[2])],degree-1,myTurtle)

myTurtle = Turtle()
myWin = myTurtle.getscreen()
myPoints = [(-300,-200),(0,200),(300,-200)]
sierpinski(myPoints,5,myTurtle)
myWin.exitonclick()
