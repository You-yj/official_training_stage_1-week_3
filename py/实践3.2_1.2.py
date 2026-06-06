from fealpy.backend import backend_manager as bm

def fd_derivative_bm(f, x0, h):
    x1 = bm.array([x0 + h] , dtype=bm.float64)
    x2 = bm.array([x0 - h] , dtype=bm.float64)
    return ((f(x1) - f(x2)) / (2*h)).item()     #.item() → 提取成原生 Python float 

def backend_set(name1, name2, f, x0, h):
    if name1 != "numpy" and name1 != "pytorch":
        print("输入后端暂时不存在")
    elif name2 != "numpy" and name2 != "pytorch":
        print("输入后端暂时不存在")
    else:
        bm.set_backend(name1) #使用 name1 后端
        print(f'a_{name1} = {fd_derivative_bm(f, x0, h)}')
        a = fd_derivative_bm(f, x0, h)
        print(f'a_{name1}的类型为： {type(a)}')
        bm.set_backend(name2) #使用 name2 后端
        print(f'a_{name2} = {fd_derivative_bm(f, x0, h)}')
        b = fd_derivative_bm(f, x0, h)
        print(f'a_{name2}的类型为： {type(b)}')
        print("误差为:",abs(a - b))              #由于上面函数用了.item()，故可以直接减
        bm.set_backend("numpy")           #这里要重新切回"numpy"后端，若不然执行这个函数后会是"pytorch"后端，导致该函数不能重复调用
        print("-"*40)

backend_set("numpy", "pytorch", bm.cos, 1.23, 1e-6)
backend_set("numpy", "pytorch", bm.cos, 2, 1e-6)
backend_set("pytorch", "numpy", bm.cos, 20, 1e-3)




# 使用 NumPy 后端
#bm.set_backend('numpy')
#print('backend numpy:', fd_derivative_bm(bm.cos, 6, 1))

# 使用 PyTorch 后端
#bm.set_backend('pytorch')
#print('backend pytorch:', fd_derivative_bm(bm.cos, 6, 1))