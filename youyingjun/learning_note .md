# 学习任务 3.1 

**完成思路**

1. **核心目标**：创建指定形状的 3D 张量并沿最后一维完成加权求和，验证对 NumPy 张量运算和爱因斯坦求和约定（einsum）的掌握。
2. **步骤拆解**：
   - 张量构造：利用`np.arange(24)`生成连续序列，通过`reshape(2,3,4)`转换为形状为 (2,3,4) 的 3D 张量，确保维度匹配要求；
   - 权重定义：手动设定长度为 4 的权重向量`w`（与张量最后一维长度一致），覆盖正、负、小数等不同类型数值，验证加权逻辑鲁棒性；
   - 加权求和：使用`np.einsum('ijk,k->ij', T, w)`实现沿最后一维（k 轴）的缩并运算，`ijk`对应 3D 张量维度，`k`对应权重向量，`->ij`指定输出为 2D 矩阵，最终输出结果并验证形状是否为 (2,3)
3. **验证逻辑**：通过打印张量、权重、结果及形状，直观确认运算维度匹配和数值计算正确性

```python
import numpy as np

#创建一个 3D 张量 T，形状为 (2, 3, 4)，用简单序列填充
T = np.arange(24).reshape(2 , 3 , 4)
print(f'T为{T},\n数据类型为{type(T)}')

#对 T 沿最后一个轴进行简单加权求和
w = np.array([1 , 3 , -2 , 0.5])#加权
M = np.einsum('ijk,k->ij', T, w) #加权求和M_{ij}=w_kT_{ijk}，遵守爱因斯坦求和约定
#shape_M = shape(M) 
print(f'对 T 沿最后一个轴进行简单加权求和，结果为:\n{M}，形状为{M.shape}')
```

## 实践任务 2

该任务的难点在与如何利用3.1学习文档中的常用的稀疏矩阵存储格式构建指定阶数的稀疏矩阵

1. 观察到任务要求的矩阵的特殊性，我采用 **COO** 稀疏矩阵存储格式

2. 我分别给定 主对角线元素、上对角线元素、下对角线元素 data、row、col(均为数组)

3. 然后再利用 `np.concatenate` 将其对应组合 成为指定稀疏矩阵的data、row、col，结合 `scipy.sparse.coo_matrix(data , (row , col)) , shape(50 , 50)`  进行存储

由于后面又要对 3 阶的稀疏矩阵求解，一个一个改数据太麻烦，我将写好的代码改为一个一个函数

**详细程序见** [实践3.1_2.py](py/实践3.1_2.py) 

**结果图如下所示：**

<div align="center">
    <img src="https://raw.githubusercontent.com/You-yj/official_training_stage_1-week_3/6a3a18cbe0bd9c41d03feeadcbd7f943548c3070/photo/Figure_3.1_2.1.png" width="350" />
    <img src="https://raw.githubusercontent.com/You-yj/official_training_stage_1-week_3/6a3a18cbe0bd9c41d03feeadcbd7f943548c3070/photo/Figure_3.1_2.2.png" width="350" />
</div>

##  实践任务 3

这个任务我是先按照要求在原程序的基础上更改、增加函数得到一个`实践3_函数版.py`，再将`实践3_函数版.py`封装成类的形式，得到`实践3_类版`，充分体会了定义类时的示例函数的定义与类外函数定义的不同，以及调用类中的方法与调用类外函数的不同

> 定义在类的外面的称之为函数，定义在类的里面的函数称之为方法

### 函数版

在`实践3.1_3_函数版.py`中，主要是在原程序的基础上

1. 新增两个函数：

   ```python
   #  函数5: u(x)
   def function(x):
       return np.exp(-x)*np.cos(2*x)
   
   #  函数6：误差
   def error (u_expr, lower, upper, x_vals, u_vals):
       return abs(float(sympy_integral(u_expr, lower, upper))-scipy_numerical_integral(x_vals, u_vals))
       #sympy 符号不能直接和浮点数做减法，符号积分要转浮点float()
   ```

2. 更改函数4

   - 在函数4中调用上述新增的两个函数并加了3个变量，目的是在图中标注数值积分值与误差(我还加了个符号积分值)

     ```python
         plt.text(3.3, 0.7, f'符号积分值为 {float(sympy_integral(u_expr, lower, upper))}', fontsize=12, color='limegreen')
         plt.text(3.3, 0.5, f'数值积分值为 {scipy_numerical_integral(x_vals, u_vals)}', fontsize=12, color='limegreen')
         plt.text(3.3, 0.3, f'误差为 {error(u_expr, lower, upper, x_vals, u_vals)}', fontsize=12, color='limegreen')
     ```

   - 填充 `u(x) > 0` 的区域（正值区域）

     ```python
     plt.fill_between(x_vals, np.zeros_like(u_vals), u_vals, where=u_vals>0, color='m', alpha=0.4, label='正值区域')
     ```

3. 还有其余的一点小修改得到最终程序

**详细程序见** [实践3.1_3_函数版.py](py/实践3.1_3_函数版.py) 

**结果图如下所示：**

<div align="center">
    <img src="https://raw.githubusercontent.com/You-yj/official_training_stage_1-week_3/6a3a18cbe0bd9c41d03feeadcbd7f943548c3070/photo/Figure_3.1_3.1.png" width="500" />
</div>

### 类版

基于`实践3.1_3_函数版.py`，我把定义函数会大部分重复出现的参数（原函数版需要反复传入的）作为类的示例参数，由于后面**x_vals**, **u_vals**需要用类中方法生成，故没将此作为示例参数

需要注意的是

1. 由于**n** , **u_expr**, **lower**, **upper**(当然还有self首参)已作为示例参数

   - 在类中**定义方法**时，参数与示例参数重复的部分用 **self** 代替，，且必须为首参；其余的独有参数保留
   - 在类中**方法内部**引用示例参数时需要 `self.示例参数名`而不能直接`示例参数名`；其余的独有参数与函数内部应用参数一致

2. 利用定义好的类创建好对象后，若要使用类中的方法

   - 如果方法的参数全与示例参数重复，即为 self ，则直接调用，不用单独传参

     例如程序第 59、63行代码

   - 如果方法的参数除 self 外还有自定义参数，调用时需要传入实参

     例如程序第66、70行代码

- 固定不变数据：`__init__存self.xxx`，方法不传参、内部 self 调用

- 运行生成数据：`x_vals/u_vals`，方法形参保留，调用实时传入

**详细程序见** [实践3 .1_3_类版.py](py/实践3 .1_3_类版.py) 

**结果图如下所示：**

<div align="center">
    <img src="https://raw.githubusercontent.com/You-yj/official_training_stage_1-week_3/6a3a18cbe0bd9c41d03feeadcbd7f943548c3070/photo/Figure_3.1_3.2.png" width="500" />
</div>
## 使用 AI 的情况

1. 作为**语法参考**

   对于没见过的语法、什么函数能先所给定功能的问题

   - 遇到陌生库语法（sympy 积分、einsum、scipy 稀疏矩阵等），查询函数用法、参数格式
   - 询问 AI 如何在图中图中标注文本注释，得出`plt.text (x,y,"文字") `定点写字，然后参考所给示例优化我的程序

2. **检查程序语法**

   直接复制粘贴所写代码，AI 会自动找出 bug，但千万不能直接复制粘贴 AI 所给的修改后的程序，我一般是看他所给的 bug 文字内容，只参考报错原因、错误位置描述

   - 在将`实践3_函数版.py`封装称类时，不时会有些位置会忘记该，比如示例方法形参忘记带 self首参，方法内部引用示例参数时忘改`self.xxx`

3. **优化文档内容**

   借助 AI 优化作业说明文字，理顺项目思路描述

# 学习任务 3.2

## fealpy常用模块

### fealpy.backend

`from fealpy.backend import backend_manager as bm`

`bm.array` 用法类似于 `np.array` ，`bm.sin`用法类似于`np.sin`，等等

**backend_manager** 相比于 numpy 、pytorch的好处是，backend_manager 

1. 可以将numpy(bm.array得到的默认为numpy后端)切换pytorch(此时均为CPU)`bm.set_backend('pytorch')`

2. 也可以切换为pytorch后将CPU切换为GPU设备`bm.set_default_device('cuda')`

### fealpy.utils

`from fealpy.utils import timer`

计时器

```python
tmr = timer()      #初始化
next(tmr)             #计时开始
。。。。                            #进行的一些操作1
tmr.send(' tag1(一般是关于上述‘进行的一些操作1’的就要说明)')       
#输出 next(tmr)  -->  tmr.send(' tag1') 的执行时间

。。。。                            #进行的一些操作2
tmr.send(' tag2(一般是关于上述‘进行的一些操作2’的就要说明)')  
#输出 tmr.send(' tag1')  -->  tmr.send(' tag2') 的执行时间

next(tmr)             #计时结束
```

### logger

`from fealpy import logger`

日志    

## 启动GPU

在实践3.2任务1.3中，我对numpy、pytorch(CPU)、pytorch(GPU)进行比较，但`bm.set_backend('pytorch')`后，再`bm.set_default_device('cuda')`无法使用pytorch(GPU)，**日志**给出：RuntimeError: No CUDA GPUs are available

于是我查看了相关学习视频：[搭建舒适的Ubuntu使用环境系列——安装显卡驱动、CUDA、cuDNN及其简单介绍](https://www.bilibili.com/video/BV16Y411M7SC/?spm_id_from=333.337.search-card.all.click&vd_source=88865534b3e9718a29e0366ce24a1efe)，问题得到解决

解决步骤：

1. 打开终端，输入`nvidia-smi`，就会显示当前驱动信息

   一开始，我显示的是：(说明我确实没有启动驱动，因为我记得我电脑是有NVIDIA GeForce MX450独显的)

   ```
   NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
   ```

2. 打开软件：Software & Updates

   选择 Additional Drivers

   我一开始是 

   ```
   Using NVIDIA driver (open kernel) metapackage from nvidia-driver-580-open (proprietary)
   ```

   改成了

   ```
   Using NVIDIA driver metapackage from nvidia-driver-580 (proprietary)
   ```

   然后应用、重启电脑

3. 再次打开终端，输入`nvidia-smi`，就会显示当前驱动信息（此时已有）

## 实践1.2

详细代码见 [实践3.2_1.2.py](py/实践3.2_1.2.py) 

### 第一种

```python
from fealpy.backend import backend_manager as bm

def fd_derivative_bm(f, x0, h):
    x1 = bm.array([x0 + h])# , dtype=bm.float64)
    x2 = bm.array([x0 - h])# , dtype=bm.float64)
    return ((f(x1) - f(x2)) / (2*h))#.item()     #.item() → 提取成原生 Python float 

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
        #print("误差为:",abs(a - b))              #由于上面函数用了.item()，故可以直接减
        bm.set_backend("numpy")            #这里要重新切回"numpy"后端，若不然执行这个函数后会是"pytorch"后端，导致该函数不能重复调用
        print("-"*40)

backend_set("numpy", "pytorch", bm.cos, 1.23, 1e-6)
backend_set("numpy", "pytorch", bm.cos, 2, 1e-6)
backend_set("pytorch", "numpy", bm.cos, 20, 1e-3)
```

该方法是直接肉眼观察得到的结果，并没有对更改后端得出的结果进行数据类型转换或提取`float`原生类型，得出的结果如下所示，误差较大

```
_numpy = [-0.9424888]
a_numpy的类型为： <class 'numpy.ndarray'>
/home/you-yingjun/Desktop/task_3/张量化计算实现/task_3_part_2/实践3.2_1.2.py:6: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  return ((f(x1) - f(x2)) / (2*h))#.item()     #.item() → 提取成原生 Python float
a_pytorch = tensor([-0.9537])
a_pytorch的类型为： <class 'torch.Tensor'>
----------------------------------------
a_numpy = [-0.90929743]
a_numpy的类型为： <class 'numpy.ndarray'>
a_pytorch = tensor([-0.8643])
a_pytorch的类型为： <class 'torch.Tensor'>
----------------------------------------
a_pytorch = tensor([-0.9124])
a_pytorch的类型为： <class 'torch.Tensor'>
a_numpy = [-0.9129451]
a_numpy的类型为： <class 'numpy.ndarray'>
----------------------------------------
```

#### 日志与排查实践

除此之外，还出现了一个DeprecationWarning日志，它有几个作用：

1. 定位代码位置

   `实践3.2_1.2.py:6`：**明确报错在第 6 行**，直接锁定你中心差分返回语句，不用全文件排查。

2. 告知风险：未来 numpy 版本会报错崩溃

   - `DeprecationWarning = 废弃警告`：**现在代码能正常运行、计算结果没错，只是弹窗提醒**
   - `Deprecated NumPy 2.0`：Numpy2 改了`__array_wrap__`接口规范，**以后新版本 numpy 会从警告→直接抛出异常、程序终止**
   - 根源：`f()`返回的不是普通 float/numpy 标量，是**自定义张量 / 重载了 numpy 数组运算的类**，这个类内部`__array_wrap__`函数写法还是 Numpy1.x 旧格式，不符合 Numpy2 新标准**(给内容根据AI得到，方法就是粘贴DeprecationWarning日志后面那段话，问他这是什么意思)**

3. 区分：不是 BUG，不影响当前计算，警告 ≠ 报错

> 该节第二种、第三种也会出现上述相同的日志，为了更好看我把它删掉

### 第二种

修改第一种的`fd_derivative_bm(f, x0, h)`函数

```python
def fd_derivative_bm(f, x0, h):
    x1 = bm.array([x0 + h] , dtype=bm.float64)
    x2 = bm.array([x0 - h] , dtype=bm.float64)
    return ((f(x1) - f(x2)) / (2*h))#.item()     #.item() → 提取成原生 Python float 
```

```
a_numpy = [-0.9424888]
a_numpy的类型为： <class 'numpy.ndarray'>
a_pytorch = tensor([-0.9425], dtype=torch.float64)
a_pytorch的类型为： <class 'torch.Tensor'>
----------------------------------------
a_numpy = [-0.90929743]
a_numpy的类型为： <class 'numpy.ndarray'>
a_pytorch = tensor([-0.9093], dtype=torch.float64)
a_pytorch的类型为： <class 'torch.Tensor'>
----------------------------------------
a_pytorch = tensor([-0.9129], dtype=torch.float64)
a_pytorch的类型为： <class 'torch.Tensor'>
a_numpy = [-0.9129451]
a_numpy的类型为： <class 'numpy.ndarray'>
----------------------------------------
```

肉眼观察，发现误差相比第一种有减小

### 第三种

修改第二种的`fd_derivative_bm(f, x0, h)`函数，用`.item()`  提取返回值为成原生 Python float，就可用在`def backend_set(name1, name2, f, x0, h):`中直接做差取绝对值得到误差

做法：在第二种去掉第4行第一个#；在第一种去掉第22行第一个#

```
a_numpy的类型为： <class 'float'>
a_pytorch = -0.9424888018638722
a_pytorch的类型为： <class 'float'>
误差为: 0.0
----------------------------------------
a_numpy = -0.9092974268543053
a_numpy的类型为： <class 'float'>
a_pytorch = -0.9092974268543053
a_pytorch的类型为： <class 'float'>
误差为: 0.0
----------------------------------------
a_pytorch = -0.9129450985712051
a_pytorch的类型为： <class 'float'>
a_numpy = -0.9129450985712051
a_numpy的类型为： <class 'float'>
误差为: 0.0
----------------------------------------
```

发现误差为零

## 实践1.3

**程序详情见** [实践3.2_1.3.py](py/实践3.2_1.3.py) 

结果：

```
/home/you-yingjun/FealPy/fealpy/fealpy/backend/pytorch_backend.py:520: UserWarning: Sparse invariant checks are implicitly disabled. Memory errors (e.g. SEGFAULT) will occur when operating on a sparse tensor which violates the invariants, but checks incur performance overhead. To silence this warning, explicitly opt in or out. See `torch.sparse.check_sparse_tensor_invariants.__doc__` for guidance.  (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:760.)
  mat = torch.sparse_coo_tensor(indices, values, size=shape)
Timer received None and paused.
=================================================
   ID       Time        Proportion(%)    Label
-------------------------------------------------
    1    142.336 [us]           0.011    阶数为1000,numpy
    2    163.078 [us]           0.013    阶数为10000,numpy
    3    845.671 [us]           0.068    阶数为100000,numpy
    4     11.672 [ms]           0.941    阶数为1000000,numpy
    5    974.763 [ms]          78.569    pytorch(set)
    6      1.861 [ms]           0.150    阶数为1000,pytorch(CPU)
    7    488.043 [us]           0.039    阶数为10000,pytorch(CPU)
    8      1.989 [ms]           0.160    阶数为100000,pytorch(CPU)
    9     12.641 [ms]           1.019    阶数为1000000,pytorch(CPU)
   10    440.359 [us]           0.035    cuda (set)
   11    230.342 [ms]          18.566    阶数为1000,pytorch(GPU)
   12    585.556 [us]           0.047    阶数为10000,pytorch(GPU)
   13    961.065 [us]           0.077    阶数为100000,pytorch(GPU)
   14      3.757 [ms]           0.303    阶数为1000000,pytorch(GPU)
=================================================
```

### 简要性能分析

#### 规模变化规律

阶数：$10^3→10^4→10^5→10^6$

1. **Numpy**耗时随规模暴涨：142μs→163μs→845μs→11.67ms
2. **PyTorch-CPU**整体增速不太稳定，同阶数普遍不如原生 Numpy；
3. **PyTorch-GPU**：**小阶 (1000) 反常极慢 (230ms)**，大阶 性能反超 CPU、Numpy

> 原因：GPU 存在**设备初始化、数据 CPU→PCIe→GPU 显存拷贝固定开销**，小计算量时通信成本盖过计算收益。

#### 性能瓶颈总结

1. **首要瓶颈：pytorch (set) 预处理**，占用近八成总耗时，远高于全部数值计算；
2. **GPU 短板：小规模任务启动开销过高**，低阶矩阵不适合上 CUDA；
3. Numpy 短板：百万阶后 CPU 串行运算效率低于 PyTorch GPU；

### 日志与排查实践

除此之外，还出现了一个UserWarning日志，PyTorch 稀疏警告日志（fealpy 内部构建稀疏矩阵触发）

```
/home/you-yingjun/FealPy/fealpy/fealpy/backend/pytorch_backend.py:520: /home/you-yingjun/FealPy/fealpy/fealpy/backend/pytorch_backend.py:520: UserWarning: Sparse invariant checks are implicitly disabled. Memory errors (e.g. SEGFAULT) will occur when operating on a sparse tensor which violates the invariants, but checks incur performance overhead. To silence this warning, explicitly opt in or out. See `torch.sparse.check_sparse_tensor_invariants.__doc__` for guidance.  (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:760.)
  mat = torch.sparse_coo_tensor(indices, values, size=shape)(Triggered internally at /pytorch/aten/src/ATen/Context.cpp:760.)
  mat = torch.sparse_coo_tensor(indices, values, size=shape)
```

它有几个作用：

1. **定位源头**：报错不在我的脚本，在 **FEALPy 源码 pytorch_backend 第 520 行**，是 FEALPy 底层创建 COO 稀疏张量触发，不是我代码写错

2. 警告并建议我：

   在调用 `torch.sparse_coo_tensor` 时，PyTorch 提示**稀疏张量的不变性检查被隐式关闭**，这可能在违反稀疏张量格式时导致内存错误（如段错误）。建议显式开启或关闭检查（通过 `torch.sparse.check_sparse_tensor_invariants` 上下文管理器或全局设置）来消除警告。

   如果想**禁用这个稀疏张量检查警告**，可以在创建张量前添加：

   ```python
   with torch.sparse.check_sparse_tensor_invariants(enable=False):
       mat = torch.sparse_coo_tensor(indices, values, size=shape)#这个是创建张量的代码行
   ```

## 问题遗留

VSCode运行正常单 import fealpy 提示 unresolved import ，有淡黄色波浪线提示，**如何消除，并且有导入提示**

