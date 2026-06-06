from fealpy.backend import backend_manager as bm
from fealpy.sparse import coo_matrix, csr_matrix
from fealpy.utils import timer
import torch

n =  bm.array([1000, 10000, 100000, 1000000])

tmr = timer()      #计时初始化
next(tmr)             #计时开始

# NumPy 后端（默认）
for i in n: 
    ni = int(i)
    nn = int(ni // 3)   ##确保 nn 为整数 
    I = bm.arange(0 , nn , 1)                          
    J = bm.arange(0 , nn , 1) 
    V = bm.arange(0, nn, 1, dtype=bm.float64) 
    A = coo_matrix((V, (I, J)), shape=(ni, ni))
    b = bm.linspace(0, 1000, ni, dtype=bm.float64)
    c = A @ b  
    tmr.send(f'阶数为{i},numpy')

bm.set_backend('pytorch')
tmr.send('pytorch(set)')
# PyTorch 后端
for i in n: 
    ni = int(i)
    nn = int(ni // 3)   ##确保 nn 为整数 
    I = bm.arange(0 , nn , 1)                          
    J = bm.arange(0 , nn , 1) 
    V = bm.arange(0, nn, 1, dtype=bm.float64) 
    #with torch.sparse.check_sparse_tensor_invariants(enable=False):
    A = coo_matrix((V, (I, J)), shape=(ni, ni))
    b = bm.linspace(0, 1000, ni, dtype=bm.float64)
    c  = A @ b  
    tmr.send(f'阶数为{i},pytorch(CPU)')

bm.set_default_device('cuda')
tmr.send('cuda (set)')
# PyTorch 后端 + GPU
for i in n: 
    ni = int(i)
    nn = int(ni // 3)   ##确保 nn 为整数 
    I = bm.arange(0 , nn , 1)                          
    J = bm.arange(0 , nn , 1) 
    V = bm.arange(0, nn, 1, dtype=bm.float64) 
    #with torch.sparse.check_sparse_tensor_invariants(enable=False):
    A = coo_matrix((V, (I, J)), shape=(ni, ni))
    b = bm.linspace(0, 1000, ni, dtype=bm.float64)
    c  = A @ b  
    tmr.send(f'阶数为{i},pytorch(GPU)')

next(tmr)             #计时结束