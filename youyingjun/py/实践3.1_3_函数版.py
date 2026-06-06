import numpy as np
import scipy.integrate  
from sympy import symbols, integrate, sin, exp, cos, simplify
import matplotlib.pyplot as plt

# 函数1: SymPy - 符号积分（输入：函数表达式、下限、上限，输出：符号积分值）
def sympy_integral(u_expr, lower, upper):
    x = symbols('x')   #符号变量
    integral = integrate(u_expr, (x, lower, upper))  #符号定积分。u_expr：符号表达式，被积函数；(x, lower, upper)：Sympy 固定格式：对x从下限→上限定积分
    return simplify(integral)  #自动化简最终表达式（合并分式、三角化简）

# 函数2: NumPy - 生成网格与函数值（输入：点数、下限、上限、函数，输出：x 值、u 值）
def numpy_grid(n, lower, upper, func):
    x_vals = np.linspace(lower, upper, n)
    u_vals = func(x_vals)
    return x_vals, u_vals

# 函数3: SciPy - 数值积分（输入：x_vals、u_vals，输出：数值积分值）
def scipy_numerical_integral(x_vals, u_vals):
    integral_num = scipy.integrate.simpson(u_vals, x_vals)  # Fully qualified
    return integral_num

# 函数4: Matplotlib - 可视化函数与积分区域（输入：u_expr, lower, upper, x_vals, u_vals，输出：显示图表）
def plot_function_with_area(u_expr, lower, upper, x_vals, u_vals):
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, u_vals, label='u = exp(-x)*cos(2x)', color='black')
    plt.fill_between(x_vals, u_vals, color='skyblue', alpha=0.8, label='积分区域') #alpha表示透明度
    plt.fill_between(x_vals, np.zeros_like(u_vals), u_vals, where=u_vals>0, color='m', alpha=0.4, label='正值区域')
    plt.title('函数 u = exp(-x)*cos(2x) 与积分区域')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.grid(True)
    plt.text(3.3, 0.7, f'符号积分值为 {float(sympy_integral(u_expr, lower, upper))}', fontsize=12, color='limegreen')
    plt.text(3.3, 0.5, f'数值积分值为 {scipy_numerical_integral(x_vals, u_vals)}', fontsize=12, color='limegreen')
    plt.text(3.3, 0.3, f'误差为 {error(u_expr, lower, upper, x_vals, u_vals)}', fontsize=12, color='limegreen')
    plt.show()

#  函数5: u(x)
def function(x):
    return np.exp(-x)*np.cos(2*x)

#  函数6：误差
def error (u_expr, lower, upper, x_vals, u_vals):
    return abs(float(sympy_integral(u_expr, lower, upper))-scipy_numerical_integral(x_vals, u_vals))
    #sympy 符号不能直接和浮点数做减法，符号积分要转浮点float()

# 实例化演示：运行整个流程
n = 100          # 小规模点数，便于理解
lower, upper = 0, 2*np.pi  # 积分区间
#func = function()   # 数值函数

# SymPy 实例：符号积分
u_expr = exp(-symbols('x'))*cos(2*symbols('x')) ##注意，就是把symbols('x')看成 x ，不能symbols('-x')或者symbols('2*x')
symbolic_int = sympy_integral(u_expr,  lower, upper)
print("符号积分值:", float(symbolic_int))  # 2.0

# NumPy 实例：网格生成
x_vals, u_vals = numpy_grid(n, lower, upper, function)

# SciPy 实例：数值积分
numerical_int = scipy_numerical_integral(x_vals, u_vals)
print("数值积分值:", numerical_int)  # 约 2.0

# Matplotlib 实例：可视化
plot_function_with_area(u_expr, lower, upper, x_vals, u_vals)





