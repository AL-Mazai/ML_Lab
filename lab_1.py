# 2. 输出空行再输出结果：1 2 3 4
print()
print("1 2 3 4")

# 3. 计算圆的周长和面积
import math

radius = int(input("请输入圆半径："))
circumference = 2 * math.pi * radius
area = math.pi * radius ** 2

print("圆的周长为:", circumference, "cm")
print("圆的面积为:", area, "平方cm")


# 4. 求直角三角形斜边长的函数
# 使用勾股定理求解
def hypotenuse(a, b):
    return math.sqrt(a ** 2 + b ** 2)
# 测试函数
print("直角三角形斜边长为:", hypotenuse(3, 4))

# 5. 使用 for 循环输出列表中的元素
num_list = [1, 2, 3, 4, 5]
for item in num_list:
    print(item)
