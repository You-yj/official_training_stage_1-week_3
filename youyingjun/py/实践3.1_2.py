import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#x
def grid(n):
    h = 1/(n+1)
    x = np.arange(h, 1, h)
    return x
    #print(f'x为{x},\nx的形状为{x.shape}。')

#A
def sparse_matrix (n):
    h = 1/(n+1)
    h2 = h**2
    diag_main = np.full(n, 2*h2)
    diag_main_up = np.full(n-1, -h2)
    diag_main_down = diag_main_up
    row_diag_main = np.arange(0 , n , 1)
    col_diag_main = np.arange(0 , n , 1)
    row_diag_main_up = np.arange(0 , n-1 , 1)
    col_diag_main_up = np.arange(1 , n , 1)
    row_diag_main_down = np.arange(1 , n , 1)
    col_diag_main_down = np.arange(0 , n-1 , 1)
    data = np.concatenate([diag_main,diag_main_up,diag_main_down])
    row =  np.concatenate([row_diag_main , row_diag_main_up , row_diag_main_down])
    col = np.concatenate([col_diag_main , col_diag_main_up , col_diag_main_down])
    return sp.sparse.coo_matrix((data, (row, col)), shape=(n,n))
    #print(A.todense())

#b
def vector (n):
    h = 1/(n+1)
    x = np.arange(h, 1, h)
    b = x*(1-x)
    return b
    #print(f'b为{b},\nb的形状为{b.shape}。')

#求解线性方程组 Ax=b
def solution (n):
    A = sparse_matrix (n)
    b = vector (n)
    u = sp.sparse.linalg.spsolve(A , b)  # 求解 u
    print(f'线性方程组的解 u 为{u}')
    return u

#可视化
def plot (n):
    fig, ax = plt.subplots() 
    x = grid(n)
    u = solution (n)
    plt.plot(x, u) 
    plt.title(f" AX=b 解的整体结构(n={n})") # 设置图表标题
    plt.xlabel("x")     # 设置 x 轴标签
    plt.ylabel("u")     # 设置 u 轴标签
    plt.grid(True)      # 显示网格
    plt.show()

n1 = 50
n2 = 3
solution (n1)
plot (n1)
solution (n2)
plot (n2)