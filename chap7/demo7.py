# 张: 学习
# 开发时间: 2021/9/2 9:43

items=['Fruits','Books','Others']
prices=[96,78,85]

d={item.upper():prices    for item,prices in zip(items,prices)}
print(d)