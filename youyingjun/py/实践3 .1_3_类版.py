import numpy as np
import scipy.integrate  
from sympy import symbols, integrate, sin, exp, cos, simplify
import matplotlib.pyplot as plt

class week_3:
    def __init__(self, n , u_expr, lower, upper):
        self.n = n
        self.u_expr = u_expr
        self.lower = lower
        self.upper = upper

    # 函数1: SymPy - 符号积分（输入：函数表达式、下限、上限，输出：符号积分值）
    def sympy_integral(self):
        x = symbols('x')   #符号变量
        integral = integrate(self.u_expr, (x, self.lower, self.upper))  #符号定积分。u_expr：符号表达式，被积函数；(x, lower, upper)：Sympy 固定格式：对x从下限→上限定积分
        return simplify(integral)  #自动化简最终表达式（合并分式、三角化简）

    # 函数2: NumPy - 生成网格与函数值（输入：点数、下限、上限、函数，输出：x 值、u 值）
    def numpy_grid(self):
        x_vals = np.linspace(self.lower, self.upper, self.n)
        u_vals = self.function(x_vals)
        return x_vals, u_vals

    # 函数3: SciPy - 数值积分（输入：x_vals、u_vals，输出：数值积分值）
    def scipy_numerical_integral(self, x_vals, u_vals):
        integral_num = scipy.integrate.simpson(u_vals, x_vals)  # Fully qualified
        return integral_num

    # 函数4: Matplotlib - 可视化函数与积分区域（输入：x_vals、u_vals，输出：显示图表）
    def plot_function_with_area(self, x_vals, u_vals):
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, u_vals, label='u = exp(-x)*cos(2x)', color='black')
        plt.fill_between(x_vals, u_vals, color='skyblue', alpha=0.8, label='积分区域') #alpha表示透明度
        plt.fill_between(x_vals, np.zeros_like(u_vals), u_vals, where=u_vals>0, color='m', alpha=0.4, label='正值区域')
        plt.title('函数 u = exp(-x)*cos(2x) 与积分区域')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.grid(True)
        plt.text(3.3, 0.7, f'符号积分值为 {float(self.sympy_integral())}', fontsize=12, color='limegreen')
        plt.text(3.3, 0.5, f'数值积分值为 {self.scipy_numerical_integral(x_vals, u_vals)}', fontsize=12, color='limegreen')
        plt.text(3.3, 0.3, f'误差为 {self.error(x_vals, u_vals)}', fontsize=12, color='limegreen')
        plt.show()

    #  函数5: u(x)
    def function(self, x):
        return np.exp(-x)*np.cos(2*x)

    #  函数6：误差
    def error (self, x_vals, u_vals):
        return abs(float(self.sympy_integral())-self.scipy_numerical_integral(x_vals, u_vals))
        #sympy 符号不能直接和浮点数做减法，符号积分要转浮点float()


v1 = week_3(100 , exp(-symbols('x'))*cos(2*symbols('x')), 0, 2*np.pi)
# SymPy 实例：符号积分

symbolic_int = v1.sympy_integral()
print("符号积分值:", float(symbolic_int))  # 2.0

# NumPy 实例：网格生成
x_vals, u_vals = v1.numpy_grid()

# SciPy 实例：数值积分
numerical_int = v1.scipy_numerical_integral(x_vals, u_vals)
print("数值积分值:", numerical_int)  # 约 2.0

# Matplotlib 实例：可视化
v1.plot_function_with_area( x_vals, u_vals)





