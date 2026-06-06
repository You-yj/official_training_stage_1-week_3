import numpy as np


#创建一个 3D 张量 T，形状为 (2, 3, 4)，用简单序列填充
T = np.arange(24).reshape(2 , 3 , 4)
print(f'T为{T},\n数据类型为{type(T)}')

#对 T 沿最后一个轴进行简单加权求和
w = np.array([1 , 3 , -2 , 0.5])#加权
M = np.einsum('ijk,k->ij', T, w) #加权求和M_{ij}=w_kT_{ijk}，遵守爱因斯坦求和约定
#shape_M = shape(M) 
print(f'对 T 沿最后一个轴进行简单加权求和，结果为:\n{M}，形状为{M.shape}')


