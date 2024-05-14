import numpy as np

#
# 2
# 2.1使用单一值创建数组
# 创建具有 10 个元素的全 0 值数组
array1 = np.zeros(10)
# 创建 2x3 的全 0 值二维数组
array2 = np.zeros((2, 3))
# 创建 2x3 的全 0 值二维整数数组
array3 = np.zeros((2, 3), dtype=int)
# 创建 2x3 的全 1 值二维数组
array4 = np.ones((2, 3))
# 创建 2x3 的二维数组，每个元素值都是 5
array5 = np.full((2, 3), 5)
# 创建 3x3 的二维数组，并且主对角线上元素都是 1
array6 = np.eye(3)
# 创建 mxn 的二维数组，并且主对角线上元素都是 1
m, n = 4, 5
array7 = np.eye(min(m, n))
# 创建 2x3 的二维数组，不指定初始值
array8 = np.empty((2, 3))

# 2.2从现有数据初始化数组
# 创建 5 个元素的一维数组，初始化为 1,2,3,4,5
array9 = np.array([1, 2, 3, 4, 5])
# 创建 2x3 的二维数组，用指定的元素值初始化
array10 = np.array([[1, 2, 3], [4, 5, 6]])
# a 是 mxn 数组，根据 a 的维度生成 mxn 的全 0 值数组 b
a = np.array([[1, 2], [3, 4]])
b = np.zeros_like(a)
# 以指定的主对角线元素创建对角矩阵
diagonal_values = [1, 2, 3]
diagonal_matrix = np.diag(diagonal_values)

# 2.3将指定数值范围切分成若干份，形成数组
# 根据指定的间距，在[m,n)区间等距切分成若干个数据点，形成数组
range_array1 = np.arange(1, 10, 2)  # 从1到10，间距为2
# 根据指定的切分点数量，在[m,n]区间等距切分成若干个数据点，形成数组
range_array2 = np.linspace(1, 10, 5)  # 从1到10，切分成5个数据点
# 生成指数间隔(而非等距间隔)的数组
exponential_array = np.logspace(0, 3, 4)  # 生成10^0到10^3，共4个数据点
# 生成网格数据点
x = np.array([1, 2, 3])
y = np.array([4, 5])
grid_x, grid_y = np.meshgrid(x, y)

# 2.4数组的引用与拷贝
# 使数组 b 与数组 a 共享同一块数据内存(数组 b 引用数组 a)
a = np.array([1, 2, 3])
b = a.view()
# 将数组 a 的值做一份拷贝后再赋给 b，a 和 b 各自保留自己的数据内存
a = np.array([1, 2, 3])
b = a.copy()


# 3
# 生成网格数据点
import matplotlib.pyplot as plt

x_values = np.linspace(0, 1, 5)  # 水平方向的数据点
y_values = np.linspace(0, 1, 3)  # 数值方向的数据点
grid_x, grid_y = np.meshgrid(x_values, y_values)

# 打印网格数据点
print("grid_x:")
print(grid_x)
print("grid_y:")
print(grid_y)

# 绘制网格
plt.scatter(grid_x, grid_y, color='red')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Grid')
plt.grid(True)
plt.show()

# 4
# 4.1使用单一值创建数组
# 创建一维数组
array1 = np.array([1, 2, 3, 4, 5, 6])
print("初始数组：")
print(array1)
# 查看数组的维度尺寸
print("数组的维度尺寸:", array1.shape)
# 一维数组变形为 mxn 二维数组
m, n = 2, 3
array2 = array1.reshape(m, n)
print("一维数组变形为 mxn 二维数组:")
print(array2)
# 将二维数组调整为一行或一列
array3 = array2.flatten()
print("将二维数组调整为一行或一列:")
print(array3)
# 行数组转成列数组
array4 = array1[:, np.newaxis]
print("行数组转成列数组:")
print(array4)
# 二维数组展成连续的一维数组
array5 = array2.ravel()
print("二维数组展成连续的一维数组:")
print(array5)
# 二维数组展成连续的一维数组(拷贝)
array6 = array2.flatten()
print("二维数组展成连续的一维数组(拷贝):")
print(array6)
# 将原有数组调整为新指定尺寸的数组(拷贝)
new_shape = (3, 2)
array7 = np.resize(array1, new_shape)
print("将原有数组调整为新指定尺寸的数组(拷贝):")
print(array7)
# 生成转置数组(矩阵)
array8 = np.transpose(array2)
print("生成转置数组(矩阵):")
print(array8)

# 4.2数组的组合、拼接及拆分
# 以竖直方向叠加两个数组
array9 = np.array([[1, 2, 3], [4, 5, 6]])
array10 = np.array([[7, 8, 9], [10, 11, 12]])
vertical_stack = np.vstack((array9, array10))
print("以竖直方向叠加两个数组:")
print(vertical_stack)
# 以水平方向叠加两个数组
horizontal_stack = np.hstack((array9, array10))
print("以水平方向叠加两个数组:")
print(horizontal_stack)
# 竖直方向将二维数组拆分成若干个数组
split_vertical = np.vsplit(vertical_stack, 2)
print("竖直方向将二维数组拆分成若干个数组:")
print(split_vertical)
# 水平方向将二维数组拆分成若干个数组
split_horizontal = np.hsplit(horizontal_stack, 2)
print("水平方向将二维数组拆分成若干个数组:")
print(split_horizontal)
# 4.3访问及修改元素
# 访问二维数组
print("访问二维数组:")
print(array9)
# 访问一维数组的部分元素
print("访问一维数组的部分元素:", array1[1:4])
# 访问二维数组的部分元素
print("访问二维数组的部分元素:")
print(array9[1, 1])
# 删除元素
array11 = np.delete(array1, [0, 2])
print("删除元素:")
print(array11)
# 删除行或列
array12 = np.delete(array9, 0, axis=0)  # 删除第一行
print("删除行:")
print(array12)
# 插入元素、行或列
array13 = np.insert(array1, 2, [10, 11])  # 在索引为2的位置插入两个元素
print("插入元素:")
print(array13)
# 追加元素、行或列
array14 = np.append(array1, [6, 7])  # 追加两个元素到末尾
print("追加元素:")
print(array14)
# 在一个二维数组后添加一列
array15 = np.array([[1, 2], [3, 4], [5, 6]])
column_to_append = np.array([7, 8, 9])
array15_with_column = np.column_stack((array15, column_to_append))
print("在一个二维数组后添加一列:")
print(array15_with_column)


# 5
# 5.1检索符合条件的元素
# 一维数组中，查找不为 0 的元素
array1 = np.array([0, 1, 2, 0, 3, 0, 4])
nonzero_elements_1d = array1[array1 != 0]
print("一维数组中，查找不为 0 的元素:", nonzero_elements_1d)

# 二维数组中，查找不为 0 的元素
array2 = np.array([[0, 1, 2], [3, 0, 4], [0, 5, 0]])
nonzero_elements_2d = array2[array2 != 0]
print("二维数组中，查找不为 0 的元素:", nonzero_elements_2d)

# 查找指定条件的元素
specified_condition = array1[array1 % 2 == 0]
print("查找指定条件的元素:", specified_condition)

# 返回条件为 True 的元素
condition_true = np.extract(array1 % 2 == 0, array1)
print("返回条件为 True 的元素:", condition_true)

# 返回指定索引的若干个元素
specified_indices = array1[[0, 2, 4]]
print("返回指定索引的若干个元素:", specified_indices)

# 5.2数组排序
# 将数组倒序
array_reverse = np.flip(array1)
print("将数组倒序:", array_reverse)

# 一维数组排序
array_sorted_1d = np.sort(array1)
print("一维数组排序:", array_sorted_1d)

# 二维数组排序
array_sorted_2d = np.sort(array2, axis=None)
print("二维数组排序:", array_sorted_2d)

# 以指定索引位置作为分界线，左边元素都小于分界元素，右边元素都大于分解元素
partitioned_array = np.partition(array1, 2)
print("以指定索引位置作为分界线:", partitioned_array)

# 5.3数组统计
# 查找一维数组中的最大、最小值
max_value_1d = np.max(array1)
min_value_1d = np.min(array1)
print("一维数组中的最大值:", max_value_1d)
print("一维数组中的最小值:", min_value_1d)

# 查找二维数组总的最大、最小值
max_value_2d = np.max(array2)
min_value_2d = np.min(array2)
print("二维数组总的最大值:", max_value_2d)
print("二维数组总的最小值:", min_value_2d)

# 查找极值元素的索引
max_index = np.argmax(array1)
min_index = np.argmin(array1)
print("一维数组中的最大值索引:", max_index)
print("一维数组中的最小值索引:", min_index)

# 统计数组中非零元素个数
nonzero_count = np.count_nonzero(array1)
print("数组中非零元素个数:", nonzero_count)

# 计算数组算数平均值
mean_value = np.mean(array1)
print("数组的算数平均值:", mean_value)

# 计算数组的加权平均值
weights = np.array([1, 2, 3, 4, 5, 6, 7])
weighted_mean = np.average(array1, weights=weights)
print("数组的加权平均值:", weighted_mean)


# 6
# 6.1数组(行向量)创建及基本运算
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([5, 4, 3, 2, 1])
print("初始向量：")
print(x1)
print(x2)

# 各元素分别计算 x1 * 5 + 2
result1 = x1 * 5 + 2
print("各元素分别计算 x1 * 5 + 2:", result1)

# 对应元素分别相乘
result2 = x1 * x2
print("对应元素分别相乘:", result2)

# 6.2创建列向量（一维数组可视为一个行向量；如果要生成列向量，必须使用 mx1 的二维数组）
# 直接创建列向量
column_vector1 = np.array([[1], [2], [3], [4], [5]])
print("直接创建列向量:")
print(column_vector1)

# 将列向量转成 1xm 的行向量
row_vector1 = column_vector1.flatten()
print("将列向量转成 1xm 的行向量:")
print(row_vector1)

# 将一个一维数组转成列向量
array1 = np.array([1, 2, 3, 4, 5])
column_vector2 = array1[:, np.newaxis]
print("将一个一维数组转成列向量:")
print(column_vector2)

# 借助 mat 类，然后通过转置实现列向量
column_vector3 = np.mat([1, 2, 3, 4, 5]).T
print("借助 mat 类，然后通过转置实现列向量:")
print(column_vector3)

# 使用 reshape 方法, 将原来的一维数组变成了 mx1 形式的二维数组
column_vector4 = array1.reshape(-1, 1)
print("使用 reshape 方法, 将原来的一维数组变成了 mx1 形式的二维数组:")
print(column_vector4)


# 7
# 7.1mat 形式的矩阵
# 创建矩阵
matrix1 = np.mat([[2, 3, 4], [5, 8, 2]])
vector1 = np.mat([[2], [1], [6]])

# 矩阵与向量乘积
result_vector = matrix1 * vector1
print("矩阵与向量乘积:")
print(result_vector)

# 矩阵与矩阵乘积
matrix2 = np.mat([[2, 3, 4], [5, 8, 2]])
matrix3 = np.mat([[2, 1], [1, 0], [6, 7]])
result_matrix = matrix2 * matrix3
print("矩阵与矩阵乘积:")
print(result_matrix)

# 7.2数组(array)形式的矩阵创建矩阵和向量（列向量、行向量）
# 生成特殊矩阵
zeros_matrix = np.zeros((2, 3))
ones_matrix = np.ones((2, 3))
identity_matrix = np.identity(3)
diag_matrix = np.diag([1, 2, 3, 4])

# 7.3关于"*"操作符
# 数组/矩阵与标量进行计算：数组/矩阵中的每个元素与标量分别计算
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([[1, 2, 3], [4, 5, 6]])
x3 = np.mat([[1, 2, 3], [4, 5, 6]])
r1 = x1 * 2
r2 = x2 * 2
r3 = x3 * 2
print(r1)
print(r2)
print(r3)

# mat 形式矩阵/向量之间的计算：完全遵照矩阵乘法进行
A1 = np.mat([[1, 2, 3], [4, 5, 6]])
A2 = np.mat([[1, 2], [3, 4], [5, 6]])
x1 = np.mat([[1], [2], [3]])
r4 = A1 * A2
r5 = A1 * x1
print(r4)
print(r5)

# 数组形式矩阵/向量之间的计算：不能当成矩阵乘法来处理，而是按照对应元素相乘
A1 = np.array([[1, 2, 3], [4, 5, 6]])
A2 = np.array([[1, 2], [3, 4], [5, 6]])
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])

r1 = A1 * x2
r2 = x2 * A1
r3 = A1 * A2
r4 = A1 * x1
r5 = x1 * A1
r6 = x1 * x2
r7 = x2 * x1

# 7.4关于 dot 函数
# 数组/矩阵与标量进行计算：数组/矩阵中的每个元素与标量分别计算，对于 a.dot(b)，若 a, b 都是一维数组，执行内积和操作
A1 = np.array([[1, 2, 3], [4, 5, 6]])
A2 = np.array([[1, 2], [3, 4], [5, 6]])
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])
x3 = np.array([1, 2])


res = []

r1 = A1.dot(A2)
res.append(r1)

r2 = A1.dot(x1)
res.append(r2)

r3 = A1.dot(x2)
res.append(r3)

# r4 = A1.dot(x3)
# r5 = x2.dot(A1)
r6 = x3.dot(A1)
res.append(r6)

print(res)


# 8
# （1）矩阵的逆
A1 = np.array([[2, 2, 3], [1, -1, 0], [-1, 2, 1]])
A1_inv = np.linalg.inv(A1)

# （2）转置矩阵
A1_transposed = A1.T

# （3）特征分解
A = np.array([[2, 2, 3], [1, -1, 0], [-1, 2, 1]])
eigen_values, eigen_vectors = np.linalg.eig(A)

# （4）SVD分解
A2 = np.array([[2, 2, 3], [4, 4, 6], [-1, 2, 1]])
U, S, VT = np.linalg.svd(A2)

# （5）矩阵的秩
A3 = np.array([[2, 1, 3], [6, 6, 10], [2, 7, 6], [1, 3, 5]])
rank_of_A3 = np.linalg.matrix_rank(A3)

# 结果
print("矩阵 A1 的逆:\n", A1_inv)
print("矩阵 A1 的转置:\n", A1_transposed)
print("矩阵 A 的特征值:\n", eigen_values)
print("矩阵 A 的特征向量:\n", eigen_vectors)
print("矩阵 A2 的 SVD 分解:\nU = ", U, "\nS = ", S, "\nVT = ", VT)
print("矩阵 A3 的秩:\n", rank_of_A3)

# 9
# 构建系数矩阵 A 和常数向量 b
A = np.array([
    [2, 1, 3],
    [6, 6, 10],
    [2, 7, 6]
])
b = np.array([2, 7, 6])

# 求解线性方程组
x = np.linalg.solve(A, b)

# 打印结果
print("方程组的解:\n", x)

# 10
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import svd

# 文档-词项矩阵
documents = np.array([
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0]
])

# 余弦相似度
cosine_sim_matrix = cosine_similarity(documents)
# 提取 d2 和 d3 之间的相似度
d2_d3_cosine_similarity = cosine_sim_matrix[1, 2]  # d2 的索引是 1，d3 的索引是 2
print("余弦相似度 (d2, d3):", d2_d3_cosine_similarity)

# SVD分解
U, S, Vt = np.linalg.svd(documents)
d2_vector = U[:, 1]  # 取第二列（索引为1）对应的特征向量
d3_vector = U[:, 2]  # 取第三列（索引为2）对应的特征向量
similarity = np.dot(d2_vector, d3_vector)  # 计算特征向量之间的内积
print('svd 计算d2和d3的相似度:', similarity)