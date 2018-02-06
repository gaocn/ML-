#Scipy使用（一）

```python
import numpy as np
from scipy import linalg, optimize

#查看方法的注释
np.info(optimize.fmin)
np.source(optimize.fmin)

```

##一、Scipy文件操作（[`scipy.io`](https://docs.scipy.org/doc/scipy/reference/tutorial/io.html)）

用于将数据导入导出成不同格式的文件。常用的是与MATLAB数据文件进行交互。

| 方法                                     | 说明                        |
| -------------------------------------- | ------------------------- |
| loadmat(file_name[,mdict, appendmat])  | 加载MATLAB数据文件              |
| savemat(file_name, mdict[, appednmat]) | 将字典或数据保存为MATLAB数据文件.mat格式 |
| whosmat(file_name[, appendant])        | 查看MATLAB文件的数据信息           |

```python
from scipy import io as sio
from numpy as np

mat_contents = sio.loadmat('octave_a.mat')
#{'a': array([[[  1.,   4.,   7.,  10.],
#        [  2.,   5.,   8.,  11.],
#        [  3.,   6.,   9.,  12.]]]),
# '__version__': '1.0',
# '__header__': 'MATLAB 5.0 MAT-file, Created on: 2013-02-17 21:02:11 UTC',
# '__globals__': []}

vect = np.arange(10)
sio.savemat('np_vect.mat', {'vect': vect})

#whosmat返回元组列表，每一数组对应一个元组(name, shape, data_type)
sio.whosmat('octave_a.mat')
#[('a', (1, 3, 4), 'double')]

#读取图片文件与matplotlib中的imread类似
#若报错，需要安装pillow包
from scipy import misc 
imdata = misc.imread("demo.jpg")
```

##二、线性代数（[`scipy.linalg`](https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html)）

###2.1 matrix VS ndarray？ 

在numpy中有两种用于矩阵计算的类，一种是numpy.matrix类，另一种是numpy.ndarray类。numpy.matrix类提供了矩阵运算常用的接口，例如: `*` 在matrix对象中就是矩阵的乘法而在ndarray对象中这不是，ndarray中需要使用a_ndarr.dot(b_ndarr)。`尽管matrix使用很方便，但是不建议使用，基于ndarray对象提供的接口更多，而引入matrix类会产生混淆！ 矩阵的所有运算一样能够使用ndarray计算出来`   

```python
import numpy as np
from scipy import linalg

#matrix对象
mat = np.mat('[1 2;3 4]')
mat.T
mat.I
mat * mat.T

#ndarray对象
A = np.array([[1,2],[3,4]])
linalg.inv(A)
b = A.T
A.dot(b)

```

###2.2 linalg基本函数

**方阵的逆：inv**

设矩阵A的逆矩阵为B，则满足AB=I且A为方阵，其中I为单位矩阵，通常$B=A^{-1}$ 

```python
import numpy as np
from scipy import linalg
A = np.array([[1,3,5],[2,5,1],[2,3,8]])
#array([[1, 3, 5],
#      [2, 5, 1],
#      [2, 3, 8]])
linalg.inv(A)
#array([[-1.48,  0.36,  0.88],
#      [ 0.56,  0.08, -0.36],
#      [ 0.16, -0.12,  0.04]])
# double check
A.dot(linalg.inv(A)) 
#array([[  1.00000000e+00,  -1.11022302e-16,  -5.55111512e-17],
#      [  3.05311332e-16,   1.00000000e+00,   1.87350135e-16],
#      [  2.22044605e-16,  -1.11022302e-16,   1.00000000e+00]])
```

**求解线性方程组：solve(A, b)**

对于线性方程组Ax = b，若方程有解即$|A| \ne 0$ 则有$x = A^{-1}b$

```python
import numpy as np
from scipy import linalg
A = np.array([[1, 2], [3, 4]])
#array([[1, 2],
#      [3, 4]])
b = np.array([[5], [6]])
#array([[5],
#      [6]])
linalg.inv(A).dot(b)  # slow
#array([[-4. ],
#      [ 4.5]])
#A.dot(linalg.inv(A).dot(b)) - b  # check
#array([[  8.88178420e-16],
#      [  2.66453526e-15]])

#方法2：速度更快
np.linalg.solve(A, b) 
array([[-4. ],
      [ 4.5]])
#check
A.dot(np.linalg.solve(A, b)) - b 
#array([[ 0.],
#      [ 0.]])
```

**行列式：det(A)**

[行列式](https://baike.baidu.com/item/%E8%A1%8C%E5%88%97%E5%BC%8F)在数学中，是一个函数，其定义域为det的矩阵A，取值为一个标量，写作det(A)或 | A | 

```python
import numpy as np
from scipy import linalg
A = np.array([[1,2],[3,4]])
linalg.det(A)
#-2.0
```

**范数：norm(a, ord=None, axis=None, keepdims=False)**

计算向量或矩阵范数，根据ord的不同norm函数能够返回不同的矩阵范数和向量范数。

- a为(M, )或者(M, N)类型的ndarray，若axis为None则a必须是1-D或2-D的数组；
- ord为范数的阶数，取值为{非零整数，inf，-inf，‘fro’}；
- axis取值为{int，2-tuple of ints，None}，若为int则表示沿哪个轴计算向量范数，若为元组则对应2-D矩阵的矩阵范数；若为None则a是1-D计算向量范数，a是2-D计算矩阵范数。
- keepdims当设置为True时， the axes which are normed over are left in the result as dimensions with size one

对于向量和矩阵对应的范数计算公式如下
$$
||X|| =
\begin{cases}
    max|X_i| \quad ord = inf \\
    min|X_i| \quad ord=-inf \\
    (\sum_{i}|X_i|^{ord})^{\frac{1}{ord}} \quad |ord| \lt \propto
\end{cases} \\


||A|| = 
\begin{cases}
    max_i\sum_j|a_{ij}| \quad ord = inf \\
    min_i\sum_j|a_{ij}| \quad ord = -inf \\
    max_j\sum_i|a_{ij}| \quad ord = 1 \\
    min_j\sum_i|a_{ij}| \quad ord = -1 \\
    max \sigma_i| \quad ord = 2 \\
    min \sigma_i| \quad ord = -2 \\
    \sqrt{trace(A^HA)} \quad ord = 'fro' \\
\end{cases}
$$

```python
import numpy as np
from scipy import linalg
A=np.array([[1,2],[3,4]])
#array([[1, 2],
#      [3, 4]])

linalg.norm(A)
#5.4772255750516612
linalg.norm(A,'fro') # frobenius norm is the default
#5.4772255750516612
linalg.norm(A,1) # L1 norm (max column sum)
#6
linalg.norm(A,-1)
#4
linalg.norm(A,np.inf) # L inf norm (max row sum)
#7    
```

**线性方程的最小二乘(least-squares)解与伪逆(Pseudo-Inverse)**

假设数据$y_i$ 与数据$x_i$ 是相关的，通过一组系数(Coefficients)$c_i$ 和函数$f_i(x_i)$ 满足模型
$$
y_i = \sum_{j}c_{j}f_{j}(x_i) + \varepsilon_i，其中\varepsilon_i 是数据中的不确定项
$$
采用最小二乘法就是找到一组系数$c_i$ 使得真实值$y_i$ 与预测值之差的平方（消除符号影响）和最小
$$
J(c) = \sum_i|y_i - \sum_{j}c_{j}f_{j}(x_i) |^2
$$
求最小值只需要分别对系数$c_i$ 求偏导数，并令偏导数等于零，求得一组解就能够使得J(c)的值最小
$$
\frac{\partial J}{\partial c_{n}^{*}} = 0 = \sum_i(y_i - \sum_{j}c_{j}f_{j}(x_i))(-f_{n}^{*}(x_i)) \\ or \\
\sum_{j}c_{j}\sum_{i}f_{j}(x_i)f_{n}^{*}(x_{i}) = \sum_{i}y_{i}f_{n}^{*}(x_i) \\
A^HAc = A^Hy \quad where \quad \{A_{ij}\} = f_{j}(x_i)  \quad and \quad  A^HA是可逆的\\
\downarrow \\
c = (A^HA)^{-1}A^Hy = A^{+}y \quad where \; 若A不可逆，则 A^{+}是A的伪逆
$$
因为$A_{ij}=f_j(x_i)$， 则定义的模型可以写为：$y = Ac + \varepsilon$ . `linalg.lstsq(A, y)`方法能够在给定A和y的情况下计算出系数c。此外l`inalg.pinv`和`linalg.pinv2`会计算出矩阵A的伪逆$A^+$

范例：下面例子使用`linalg.lstsq(A, y)` 和`inalg.pinv` 解决数据拟合问题(data-fitting)，下面数据是基于模型，其中给$y_i$ 添加了噪音
$$
y_i = c_1e^{-x_i} + c_2x_i \quad where \ x_{i} = 0.1i \ ，i \in [1, 10]  \ c_1 = 5，c_2 = 4  
$$

```python
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

c1, c2 = 5.0, 2.0
i = np.r_[1:11]
#array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
xi = 0.1*i
#array([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ])
yi = c1*np.exp(-xi) + c2*xi
#array([ 4.72418709,  4.49365377,  4.3040911 ,  4.15160023,  4.0326533 ,
#        3.94405818,  3.88292652,  3.84664482,  3.8328483 ,  3.83939721])
#带噪声的y值
zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi))

A = np.c_[np.exp(-xi)[:, np.newaxis], xi[:, np.newaxis]]
#array([[ 0.90483742,  0.1       ],
#       [ 0.81873075,  0.2       ],
#       [ 0.74081822,  0.3       ],
#       [ 0.67032005,  0.4       ],
#       [ 0.60653066,  0.5       ],
#       [ 0.54881164,  0.6       ],
#       [ 0.4965853 ,  0.7       ],
#       [ 0.44932896,  0.8       ],
#       [ 0.40656966,  0.9       ],
#       [ 0.36787944,  1.        ]])

# lstsq返回值
# c:        最小二乘的解，为(N,)或(N, K)ndarray
# residues：残差和，squared 2-norm for each column in ``y-Ac``
# rank:     矩阵A的秩
# sigma:    矩阵A的奇异值
c, resid, rank, sigma = linalg.lstsq(A, zi)
#(array([ 5.00278791,  2.10108782]),
# 0.47110363682170298,
# 2,
# array([ 2.58763467,  1.02933937]))

#用模型预测的值
xi2 = np.r_[0.1:1.0:100j]
yi2 = c[0]*np.exp(-xi2) + c[1]*xi2

#查看拟合效果
plt.plot(xi,zi,'x',xi2,yi2)
plt.axis([0,1.1,3.0,5.5])
plt.xlabel('$x_i$')
plt.title('Data fitting with linalg.lstsq')
plt.show()
```

**伪逆矩阵（广义逆矩阵）** 

| 伪逆矩阵方法       | 说明                              |
| ------------ | ------------------------------- |
| linalg.pinv  | Moore-Penrose利用最小二乘法lstsq求解伪逆矩阵 |
| linalg.pinv2 | 利用SVD分解求伪逆矩阵                    |
| linalg.pinvh | 求Hermitian矩阵的伪逆矩阵               |

奇异矩阵或非方阵的矩阵不存在逆矩阵，但可以用函数`linalg.pinv` or `linalg.pinv2` 求矩阵的伪逆矩阵。令A为M*N的矩阵，称矩阵G是A的广义逆矩阵（伪逆矩阵）。
$$
A为M \times N的矩阵，\qquad \qquad \qquad\qquad \qquad \qquad\qquad \qquad \qquad\\
1. 若M > N，则A的广义逆矩阵为：A^+ = (A^HA)^{-1}A^H  \\
2. 若M < N，则A的广义逆矩阵为：A^\# =A^H (AA^H)^{-1}   \\
3. 若M = N，则A的广义逆矩阵为：A^\# =A^+ = A^{-1}  \quad
$$

###2.3 特征值（Eigenvalues）、特征向量（Eigenvectors）

对于矩阵A存在常数$\lambda$以及对应向量v，满足等式$Av = \lambda v$ ，对于$N \times N$的矩阵存在N个特征值（不一定不同），满足多项式的$|A - \lambda I| = 0$根。特征向量v也称为右特征向量，区别于左特征向量$v_{L}^{H}$ ，左特征向量满足
$$
\ v{L}^{H} A = \lambda v{L}^{H} \quad or \quad  A^H v{L} = \lambda^{*} v{H} ，其中H表示共轭向量|矩阵（Hermitian conjugation），\lambda^{*}是\lambda的共轭向量
$$
此外，`linalg.eig` 方法能够解决广义的特征值问题，对于方阵A，B，满足
$$
A v = \lambda B v  \qquad  \qquad \ \ \ 右特征向量\\
A^H v_{L} = \lambda^{*}B^{H}v_{L}   \qquad 左特征向量
$$
同样可以求出对应的特征值和特征向量。实际上，标准的特征值和特征向量是在B=I的情况下求出的。当求解出特征向量后，我们就能够得到矩阵A的一个分解
$$
A = B V  \Lambda V_{-1} \qquad V是列特征向量构成矩阵，\Lambda是有特征值构成的对角矩阵
$$
根据定义，特征向量只取决于常量因子$\lambda$ ，在Scipy中特征向量的的常量因子满足
$$
||v||^2 = \sum_{i}v_{i}^2 = 1
$$
`范例` ：
$$
A = \left [
\begin{matrix}
    1 & 5 & 2 \\
    2 & 4 & 5 \\
    3 & 6 & 2
\end{matrix}
\right ] \\
|A-\lambda I| = (1 - \lambda)[(4 - \lambda)(2 - \lambda)-6] - 5[2(2-\lambda) - 2]  \\ 
+ 2[12 - 3(4 - \lambda)]  \qquad \qquad \\
= - \lambda^3 + 7\lambda^2 + 8\lambda - 3 \qquad \qquad \qquad \  \\
求得： \lambda_{1} = 7.9579，\lambda_{2} = -1.2577，\lambda_{3}=0.2997  \qquad  \quad
$$
**linalg.eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True)**

**linalg.eigvals(a, b=None, overwrite_a=False, check_finite=True) 仅仅返回特征值w**

- a要计算特征值和特征向量的矩阵
- b为在计算广义特征值和特征向量时，满足Av = xBv中的矩阵B
- left为True表示计算左特征向量，right默认为True表示返回右特征向量
- overwrite_a是否覆盖a矩阵，overwrite_b是否覆盖b矩阵，为False会提高计算速度；
- check_finite为True是会检查矩阵中的数是不是finite number，设置为False会提高计算速度

​     `返回值` 

- w, 类型为特征值构成(M, )的ndarray
- vl，规范化的左特征向量(M, M)的ndarray，当left=True是返回；
- vr，规范化的右特征向量(M, M)的ndarray，当right=True是返回；

```python
import numpy as np
from scipy import linalg
A = np.array([[1, 2], [3, 4]])
la, v = linalg.eig(A)
l1, l2 = la
# eigenvalues
#(-0.372281323269+0j) (5.37228132327+0j)

v
#array([[-0.82456484, -0.41597356],
#       [ 0.56576746, -0.90937671]])
# first eigenvector
print(v[:, 0])   
[-0.82456484  0.56576746]
 # second eigenvector
print(v[:, 1])  
[-0.41597356 -0.90937671]

# eigenvectors are unitary
print(np.sum(abs(v**2), axis=0))  
[ 1.  1.]

v1 = np.array(v[:, 0]).T
#check the computation
print(linalg.norm(A.dot(v1) - l1*v1)) 
3.23682852457e-16
```

###2.4 矩阵分解(Decomposition)

**奇异值分解（SVD，Singular Value Decomposition）** 

奇异值分解可以看成是特征值问题的扩展，`针对非方阵` ,  设A为$M \times N$ 的矩阵，满足$A^HA、AA^H分别是N \times N、M \times M 的 Hermitian方阵$ ，Hermitian方阵的特征值是`实数且是非负的`，并且在Hermitian方阵$A^HA、AA^H$中`最多有min(M, N)个相同的非零特征值` 。设这些非负特征值为$\sigma_{i}^2$ ，那么这些特征值开根号后的即为矩阵A的奇异值，矩阵$A^HA$ 的列特征向量构成了$N \times N$的unitary矩阵$V $ ，而矩阵$AA^H$ 的列特征向量构成的$M \times M$ 的unitary矩阵U，奇异值构成的$M \times N$对角矩阵$\Sigma$  ，则：
$$
A = U \Sigma V^H \\
 hermitian \  matrix \ D \ 满足 D^H = D \\
 unitary \ matrix \ D \ 满足 D^HD = I = DD^H \rightarrow D^{-1} = D^H
$$
就是A的一个奇异分解。每一个矩阵都存在一个奇异值分解。有时候奇异值称为矩阵的A的谱(spectrum)。`linalg.svd` 返回$U，\sigma_{i} 数组，V^H $ ，`linalg.diagsvd` 返回奇异值构成的对角矩阵$\Sigma$ `linalg.svdvals` 仅仅返回矩阵的奇异值。

```python
import numpy as np
from scipy import linalg
A = np.array([[1,2,3],[4,5,6]])
#array([[1, 2, 3],
#      [4, 5, 6]])

M,N = A.shape
U,s,Vh = linalg.svd(A)
Sig = linalg.diagsvd(s,M,N)

U, Vh = U, Vh
#array([[-0.3863177 , -0.92236578],
#      [-0.92236578,  0.3863177 ]])
#Sig
#array([[ 9.508032  ,  0.        ,  0.        ],
#      [ 0.        ,  0.77286964,  0.        ]])
#Vh
#array([[-0.42866713, -0.56630692, -0.7039467 ],
#      [ 0.80596391,  0.11238241, -0.58119908],
#      [ 0.40824829, -0.81649658,  0.40824829]])

U.dot(Sig.dot(Vh)) #check computation
#array([[ 1.,  2.,  3.],
#      [ 4.,  5.,  6.]])
```

**LU分解**

对于矩阵$M \times N $ 的A，LU分解后得到
$$
A = P \ L \ U  \\
其中P是M \times M的permutation \ matrix（单位矩阵按行随机排列得到的矩阵） \\
L是M \times K的下三角矩阵[K = min(M, N)]  \\
U是K \times N的上三角矩阵
$$
LU分解通常用于解决simultaneous equations，并且等式左边不变而右边经常变动
$$
Ax_i = b_i，（对于不同的b_i） \\
\downarrow \\
PLUx_i = b_i
$$

>An initial time spent factoring A allows for very rapid solution of similar systems of equations in the future. If the intent for performing LU decomposition is for solving linear systems then the command `linalg.lu_factor` should be used followed by repeated applications of the command `linalg.lu_solve` to solve the system for each new right-hand-side.

**linalg.lu(a, permute_l=False, overwrite_a=False, check_finite=True)**

- 默认返回值为P， L，U
- 若permute_l为True，则返回pl，U

**linalg.lu_factor(a, overwrite_a=False, check_finite=True)**

- 计算矩阵A 的pivoted LU分解
- 返回值为N*N的矩阵LU和N长度的数组piv

**linalg.lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True)**

- lu_and_piv为元组(LU， piv)
- b为Ax=b方程组中矩阵b
- trans取值为{0, 1, 2}分别对应：a x   = b， a^T x = b，a^H x = b
- 返回值为x数组即方程组的解

**Cholesky decomposition**

Cholesky分解是LU分解的一个特例，针对Hermitian矩阵（$A=A^H \ and \ x^H A x \ge 0 \ for \ all \ x$），则A的分解可以写为
$$
A = U^H U  \qquad (U是上三角矩阵)\\
A = L L^H  \qquad (L是下三角矩阵，L = U^H)
$$
同样存在方法`linalg.cholesky` ，` linalg.cho_factor`，`linalg.cho_solve`

**QR分解**

QR分解又称为a polar decomposition，对于任意$M \times N$ 的矩阵能够找到$M \times N$ 的unitary矩阵Q和$M \times N$ 上三角矩阵R，满足
$$
A = QR \\
若A的SVD分解是已知的，则有 A = U \Sigma V^H = QR，则有Q = U  \ ， R = \Sigma V^h
$$

| QR分解相关方法                     | 说明                                       |
| ---------------------------- | ---------------------------------------- |
| linalg.qr(a)                 | 返回Q, R, P                                |
| linalg.qr_multiply(a, c)     | 进行QR分解，并将Q乘以c返回，返回Qc, R, P               |
| linalg.qr_update(Q, R, u, v) | If ``A = Q R`` is the QR factorization of ``A``, return the QR factorization of ``A + u v**T`` for real ``A`` or ``A + u v**H`` for complex ``A`` ，返回Q1, R1 |
| linalg.qr_delete             | If ``A = Q R`` is the QR factorization of ``A``, return the QR  factorization of ``A`` where ``p`` rows or columns have been removed starting at row or column ``k``. |
| linalg.qr_insert             | If ``A = Q R`` is the QR factorization of ``A``, return the QR factorization of ``A`` where rows or columns have been inserted starting at row or column ``k``. |

## 三、优化方法(scipy.optimize)

###3.1 多元非约束优化方法

scipy.optimize包为约束优化和非约束优化算法提供一个公共接口`minimize` ，只需要在调用该函数时指定`method` 参数值就可。下面例子以求解函数
$$
f(x) = \sum_{i=1}^{N-1} 100 (x_i - x_{i-1}^2)^2 + (1 - x_{i-1})^2
$$
为例子求解该函数的最优解，同时给出如何求函数的雅克比矩阵(Jacobian)和海森矩阵(Hessian)的例子。

**1. Nelder-Mead Simplex 算法**

simplex算法仅仅需要输入求解函数就能求出其最优解，由于simplex算法本身没有使用任何梯度，因此求解的时间会比较长一点。此外，Powell方法也是仅根据输入函数就能求出最优解。simplex算法通常适用于简单优化问题。

```python
import numpy as np
from scipy.optimize import minimize

def rosen(x):
  # The Rosenbrock functon
  return sum(100.0 * ((x[1:] - x[:-1] ** 2.0)) ** 2.0 + (1 - x[:-1]) ** 2.0)
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead', options={'disp': True})
#Optimization terminated successfully.
#         Current function value: 0.000066
#         Iterations: 141
#         Function evaluations: 243
#final_simplex: (array([[ 0.99910115,  0.99820923,  0.99646346,  0.99297555,  0.98600385],
#       [ 0.99909442,  0.99820319,  0.99645812,  0.99302291,  0.98608273],
#       [ 0.99908478,  0.99820409,  0.9964653 ,  0.99296511,  0.9859391 ],
#       [ 0.99909641,  0.99824295,  0.99644354,  0.9929541 ,  0.98596017],
#       [ 0.99907389,  0.998211  ,  0.99645267,  0.99294537,  0.98597485],
#       [ 0.99907893,  0.99819173,  0.99644522,  0.99298154,  0.986004  ]]), array([  6.61748171e-05,   6.64266969e-05,   6.66640269e-05,
#         6.69424827e-05,   6.70671859e-05,   6.70870519e-05]))
 #          fun: 6.6174817088845322e-05
 #      message: 'Optimization terminated successfully.'
 #         nfev: 243
 #          nit: 141
 #       status: 0
 #      success: True
 #            x: array([ 0.99910115,  0.99820923,  0.99646346,  0.99297555,  0.98600385])
  
  res = minimize(rosen, x0, method='powell', options={ 'disp': True})
  #   direc: array([[ -5.04050381e-07,  -2.91450372e-06,  -3.73102198e-06,
  #       -6.68348509e-06,  -1.30888161e-05],
  #     [ -2.99937619e-03,  -5.44579148e-03,  -1.15336808e-02,
  #       -2.32125403e-02,  -4.51404333e-02],
  #     [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
  #        0.00000000e+00,   0.00000000e+00],
  #     [ -6.53046464e-03,  -4.12908316e-03,  -1.29127384e-03,
  #        5.08867880e-05,   1.21533142e-04],
  #     [ -1.42440408e-08,  -2.89907128e-08,  -5.68480023e-08,
  #       -1.16937235e-07,  -2.75813141e-07]])
  #   fun: 1.6967633615782998e-22
 #message: 'Optimization terminated successfully.'
 #   nfev: 1084
 #    nit: 18
 # status: 0
 #success: True
 #      x: array([ 1.,  1.,  1.,  1.,  1.])
```

**2. BFGS算法(Broyden-Fletcher-Goldfarb-Shanno)**

为了快速收敛，BFGS算法（梯度下降优化算法）使用目标函数的梯度。若梯度函数没有给出，默认BFGS会使用目标函数的一阶微分方程。上述例题中的梯度计算公式为
$$
\frac{\partial f}{\partial x_j} = \sum_{i=1}^{N} 200(x_i - x_{i-1}^2)( \delta_{i,j} - 2x_{i-1} \delta_{i-1,j}) - 2(1-x_{i-1}) \delta_{i-1,j} \\
=200(x_j - x_{j-1}^2) - 400x_j(x_{j+1} - x_j^2) - 2(1 - x_j) \\
两个特例：
\frac{\partial f}{\partial x_0} = -400 x_0(x_1 -x_0^2) - 2(1 - x_0) \\ 
\frac{\partial f}{\partial x_{N-1}} = 200(x_{N-1} - x_{N-2}^2)
$$

```python
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 -xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der
res = minimize(rosen, x0, method='BFGS', jac=rosen_der,  options={'disp': True})
#Optimization terminated successfully.
#         Current function value: 0.000000
#         Iterations: 25
#         Function evaluations: 30
#         Gradient evaluations: 30
#In [109]: res
#Out[109]:
#      fun: 4.0130879949972905e-13
# hess_inv: array([[ 0.00758796,  0.01243893,  0.02344025,  0.04614953,  0.09222281],
#       [ 0.01243893,  0.02481725,  0.04712952,  0.09298607,  0.18569385],
#       [ 0.02344025,  0.04712952,  0.09456412,  0.18674836,  0.37282072],
#       [ 0.04614953,  0.09298607,  0.18674836,  0.37383212,  0.74621435],
#       [ 0.09222281,  0.18569385,  0.37282072,  0.74621435,  1.49444705]])
#      jac: array([ -5.68982937e-06,  -2.73296557e-06,  -2.54520599e-06,
#        -7.73460770e-06,   5.78142698e-06])
#  message: 'Optimization terminated successfully.'
#     nfev: 30
#      nit: 25
#     njev: 30
#   status: 0
#  success: True
#        x: array([ 1.00000004,  1.0000001 ,  1.00000021,  1.00000044,  1.00000092])
```

**3. 牛顿共轭梯度算法(Newton-Conjugate-Gradient)**

牛顿共轭梯度算法是牛顿算法[^3]的改进，采用共轭梯度法求海森矩阵(Hessian)的逆。牛顿方法基于泰勒展开将本地函数转换为二次型
$$
f(x) \approx  f(x_0) + \triangledown f(x_0) \cdot (x - x_0) + \frac{1}{2}(x - x_0)^TH(x_0)(x-x_0)
$$
其中$H(x_0)$ 是函数f(x)的二阶导数得到矩阵。若海森矩阵提供了，则可以将上述梯度置为0求出迭代优化方程
$$
x_{opt} = x_0 - H^{-1} \triangledown f，其中\triangledown f = Jacobian矩阵，H为Hessian矩阵
$$
计算海森矩阵的逆采用共轭梯度(Conjugate-Gradient)。因此若要使用Newton-CG方法需要提供计算海森矩阵的方法作为参数。在optimize方法中通过`hess` 提供计算海森矩阵或提供海森矩阵与任意向量的乘积的函数。例如
$$
Jac_j = \frac{\partial f}{\partial x_j} = \sum_{i=1}^{N} 200(x_i - x_{i-1}^2)( \delta_{i,j} - 2x_{i-1} \delta_{i-1,j}) - 2(1-x_{i-1}) \delta_{i-1,j} \\
=200(x_j - x_{j-1}^2) - 400x_j(x_{j+1} - x_j^2) - 2(1 - x_j) \\
=200(x_i - x_{i-1}^2) - 400x_i(x_{i+1} - x_i^2) - 2(1 - x_i)，其中i \in[1, N-2] \\
\quad \\
H_{ij} = \frac{\partial^2 f }{\partial x_i \partial x_j} = 200(\delta_{i,j} - 2x_{i-1} \delta_{i-1, j}) - 400 \delta_{i,j}(x_{i+1} - x_i^2) - 400x_i(\delta_{i+1,j} - 2x_i \delta_{i,j}) +2 \delta_{i,j} \\
= (202 + 1200 x_i^2 - 400 x_{i+1})  \delta_{i,j} - 400x_i \delta_{i+1,j} - 400x_{i-1} \delta_{i-1,j} \quad 其中i,j \in [1, N-2]\\
$$
如果$i,j \in [0, N-1]$ 且海森矩阵为$N *N$ 则海森矩阵的其他非空条目为
$$
\frac{\partial^2 f}{\partial x_0^2} = 1200x_0^2 - 400x_1 + 2 \\
\frac{\partial f}{\partial x_0 x_1} = \frac{\partial f}{\partial x_1 x_0} = -400x_0 \\
\frac{\partial f}{\partial x_{N-1} x_{N-2}} = \frac{\partial f}{\partial x_{N-2} x_{N-1}} = -400x_{N-2} \\
\frac{\partial f}{\partial x_{N-1}^2} = 200
$$
例如，当N=5时，海森矩阵为
$$
H = 
\left [ \begin{matrix}
1200x_0^2-400x_1+2 & -400x_0 & 0 & 0 &0 \\
-400x_0 & 202 + 1200x_1^2-400x_2 & -400x_1 & 0 & 0 \\
0 & -400x_2 & 202+1200x_2^2-400x_3 & -400x_2 & 0 \\
0 & 0 & -400x_2 & 202+1200x_3^2-400x_4 & -400x_3 \\
0 & 0 & 0 & -400x_3 & 200
\end{matrix} \right]
$$

```python
def rosen_hess(x):
     x = np.asarray(x)
     H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
     diagonal = np.zeros_like(x)
     diagonal[0] = 1200*x[0]**2-400*x[1]+2
     diagonal[-1] = 200
     diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
     H = H + np.diag(diagonal)
     return H
res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess,     options={'disp': True})
#Optimization terminated successfully.
#         Current function value: 0.000000
#         Iterations: 21
#         Function evaluations: 30
#         Gradient evaluations: 50
#         Hessian evaluations: 21
#Out[131]:
#     fun: 1.859665471986481e-08
#     jac: array([  5.86892970e-05,   1.54121807e-04,   6.93311785e-04,
#         3.04308913e-03,  -1.87051318e-03])
# message: 'Optimization terminated successfully.'
#    nfev: 30
#    nhev: 21
#     nit: 21
#    njev: 50
#  status: 0
# success: True
#       x: array([ 0.9999852 ,  0.9999705 ,  0.99994098,  0.99988173,  0.99976293])
```

对于复杂方程的最小化问题，使用Hessian矩阵会消耗大量时间和内存。而使用Newton-CG可以不给出Hessian矩阵而给出Hessian矩阵与任意向量的积即可。通过`hessp` 关键字参数提供一个函数，函数返回Hessian矩阵与任意向量的积，这种计算方式可能是最快计算函数最小值的方式。假设p为任意向量，则有
$$
H(x) \cdot p = 
\left[ \begin{matrix}
(1200x_0^2 - 400x_1 + 2)p_0 - 400x_0 p_1 \\
.\\
.\\
-400x_{i-1}p_{i-1} + (202 + 1200x_i^2 - 400x_{i+1})p_i - 400x_i p_{i+1}\\
.\\
.\\
-400x_{N-2} p_{N-2} + 200p_{N-1}\\
\end{matrix} \right]
$$


```python
def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200 * x[0] ** 2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
    Hp[1:-1] = -400 * x[:-2] * p[-2] + (202 + 1200 * x[1:-1] ** 2 - 400 *x[2:]) * p[1:-1] - 400 * x[1:-1] * p[2:]
    Hp[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]
    return Hp
res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hessp=rosen_hess_p,options={'disp': True})
#Warning: Warning: CG iterations didn't converge.  The Hessian is not positive definite.
#         Current function value: 0.028015
#         Iterations: 10
#         Function evaluations: 11
#         Gradient evaluations: 21
#         Hessian evaluations: 118
```

**4. Trust-Region Newton-Conjugate-Gradient**

 使用Newton-CG算法时，若构建的Hessian矩阵质量较差，如上面例子所示，那么搜索方向就会有问题。Newton-CG是`线性搜索方法（Linear search）` ，它首先会找到能使目标函数而泰勒近似方程最小化的一个方向，然后利用线性搜索算法找到该方向上的最优步长(Step size)对结果进行迭代。

另一个方法是首先固定步长上限为$\triangle$ ，然后在这个给定trust-radius范围内找到最优步长p，通过求解下面子问题求出p值
$$
min_{p}f(x_k) + \triangledown f(x_k) \cdot p + \frac{1}{2}p^TH(x_k)p \\
subject \ to： ||p|| \le \triangle
$$
求出p值后，根据下面等式迭代x，并且根据泰勒展开式与实际函数的统一程度更新trust-radius $\triangle$ 的值。
$$
x_{k+1} = x_k + p
$$
该方法称为trust-region，而 trust-ncg算法使用共轭梯度求解 trust-region的子问题。

```python
res = minimize(rosen, x0, method='trust-ncg',jac=rosen_der, hess=rosen_hess,options={'disp': True})
#Optimization terminated successfully.
#         Current function value: 0.000000
#         Iterations: 18
#         Function evaluations: 19
#         Gradient evaluations: 18
#         Hessian evaluations: 17
# message: 'Optimization terminated successfully.'
#    nfev: 19
#    nhev: 17
#     nit: 18
#    njev: 18
#  status: 0
# success: True
#       x: array([ 0.99999696,  0.99999391,  0.99998779,  0.99997552,  0.99995092])

res = minimize(rosen, x0, method='Newton-CG',jac=rosen_der, hessp=rosen_hess_p,options={'disp': True})
#Warning: Warning: CG iterations didn't converge.  The Hessian is not positive definite.
#         Current function value: 0.028015
#         Iterations: 10
#         Function evaluations: 11
#         Gradient evaluations: 21
#         Hessian evaluations: 118
```

**5. Trust-Region Truncated Generalized Lanczos / Conjugate Gradient Algorithm **

与`trust-ncg`问题类似，`trust-krylov` 方法同样适用于large-scale问题，它同样使用Hessian矩阵作为线性算子(Linear Operator)通过计算矩阵向量的乘积得出结果。`trust-krylov` 比`trust-ncg` 方法能够得到下面问题更精确的解
$$
min_{p}f(x_k) + \triangledown f(x_k) \cdot p + \frac{1}{2}p^TH(x_k)p \\
subject \ to： ||p|| \le \triangle
$$
该方法通过对GLTR方法[^2] 的TRLIB[^1]实现进行封装，在`truncated Krylov subspace`条件下，用于解决精确解决`trust-region` 的子问题。对于`indefinite problem` 问题，与`trust-ncg`相比，使用`trust-krylov` 方法能够减少非线性迭代次数，这样每次子问题的求解就会少计算几次矩向量乘积。

```python
# Hessian
res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hess=rosen_hess, options={'disp': True})

#Hessian product
res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': True})
```

**6. Trust-Region Nearly Exact Algorithm**

方法`Newton-CG`, `trust-ncg` 和 `trust-krylov` 都适用于large-scale问题(包含上千变量)的求解。因为共轭梯度算法不需要显示对Hessian矩阵进行分解而通过近似求解Hessian的逆或求解 trust-region的子问题对问题进行迭代。由于可以通过提供Hessian product的方式进行计算，因此算法适用于处理稀疏矩阵，能够降低存储需求同时节省计算时间。

对于medium-size问题，Hessian矩阵的分解和存储需求不是很大，因此可以通过精确求解`trust-region` 子问题的方式而只需要几步迭代即可。为此，CGT[^4] 方法能够迭代求解非线性等式的泰勒二次展开式的子问题。CGT方法一般只需要3~4次Hessian矩阵的分解就可得出结果。因此与其他`trust-region` 的其他实现相比，CGT方法能够在更少的迭代次数、更少的目标函数计算次数内收敛。

```python
res = minimize(rosen, x0, method='trust-exact',jac=rosen_der, hess=rosen_hess,options={'disp': True})
#Optimization terminated successfully.
#         Current function value: 0.000000
#         Iterations: 13                    # may vary
#         Function evaluations: 14
#         Gradient evaluations: 13
#         Hessian evaluations: 14
```



###3.2 多元约束优化方法

**1. Sequential Least SQuares Programming optimization algorithm** 

序列最小二乘规划优化算法用于解决约束优化问题，它能够解决下面形式的问题
$$
min \quad F(x) \\
s.t. \quad H_i(x)，i = 1, 2,...,MEQ \\
 \qquad  \qquad  G_i \ge 0，i=MEQ+1,...,M \\
x \in [XL, XU]，
$$
例如，考虑下面约束优化问题
$$
max \quad f(x,y) = 2xy + 2x - x^2 - 2y^2 \\
s.t. \quad x^3-y = 0, \quad  y-1 \ge 0
$$
使用方法如下

```python
import numpy as np
from scipy.optimize import minimize
#目标函数
# 因为minimize函数仅仅求最小化函数，因此若求最大值需要将sign=-1.0
def obj_func(x, sign=1.0):
    return sign * (2 * x[0] * x[1] + 2 * x[0] - x[0] ** 2 - 2 * x[1] ** 2)

#目标函数的一阶导数
# 因为minimize函数仅仅求最小化函数，因此若求最大值需要将sign=-1.0
def obj_func_deriv(x, sign=1.0):
    #Derivative of objective function
    dfx0 = sign * (-2 * x[0] + 2 * x[1] + 2)
    dfx1 = sign * (2 * x[0] - 4 * x[1])
    return np.array([dfx0, dfx1])

#约束条件用于dict的列表表示，每一个约束用dict表示，每一个约束条件有"type、fun、jac"表示
cons = (
{
    'type': 'eq',
    'fun' : lambda x: np.array([x[0] ** 3 - x[1]]),
    'jac' : lambda x: np.array([3.0 * (x[0] ** 2.0), -1.0])
},
{
    'type': 'ineq',
    'fun' : lambda x: np.array([x[1] - 1]),
    'jac' : lambda x: np.array([0.0, 1.0])
}
)
x0 = [-1.0, 1.0]
# args参数是传递给obj_func和jac函数的参数
res = minimize(obj_func, x0, args=(-1.0,), jac=obj_func_deriv, constraints=cons, method='SLSQP', options={'disp': True})

#Optimization terminated successfully.    (Exit mode 0)
#            Current function value: -1.00000018311
#            Iterations: 9
#            Function evaluations: 14
#            Gradient evaluations: 9
#Out[162]:
#     fun: -1.0000001831052137
#     jac: array([-1.99999982,  1.99999982,  0.        ])
# message: 'Optimization terminated successfully.'
#    nfev: 14
#     nit: 9
#    njev: 9
#  status: 0
# success: True
#       x: array([ 1.00000009,  1.        ])
```

###3.3 Equation(local) Minimizers

**1. Least-squares minimization**

对于有界约束的非线性最小二乘问题
$$
min_{x} \  \frac{1}{2} \sum_{i=1}^{m} \rho(f_i(x)^2) \\
subject \ to \ lb \le x \le ub
$$
其中$f_i(x)$ 为平滑函数(Smooth function)，$R^n \rightarrow R$ ，称此函数为残差(residuals)。

>a scalar valued function ρ(⋅) is to reduce the influence of outlier residuals and contribute to robustness of the solution, we refer to it as a loss function

损失函数是衡量损失和错误程度的函数。若$\rho$ 为线性损失函数我们就得到标准的最小二乘问题。least-squares minimization方法需要一个f(x)的一阶偏导函数构成的M*N的矩阵，称为雅克比矩阵(Jacobian)，在使用该方法时最好显示给出，否则 [`least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares) 方法会自己通过微分求解，会消耗大量时间且结果不一定准确。

 [`least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares) 同样可以用于函数的拟合，将$\varphi(t;x)$ 应用于经验数据$\{(t_i,y_i)，i=0,1...,m-1\}$求解x，需要体检计算残差函数$f_i(x) = w_i(\varphi(t_i;x) - y_i)$ ，其中$w_i$ 为每一次预测偏差的权值。

例如，考虑一下残差函数
$$
f_i(x) = \frac{x_0(u_i^2 + u_i x_1)}{u_i^2 + u_ix_2 + x_3} - y_i，i=0,1,...,10
$$
其中$y_i$ 为测量值，$u_i$ 为独立变量值，未知向量为$x=(x_0,x_1,x_2,x_3)^T$ 。则求得残差方程的Jacobian矩阵为
$$
J_{i0} = \frac{\partial f_i}{\partial x_0} = \frac{u_i^2 + u_i x_1}{u_i^2 + u_ix_2 + x_3} \\
J_{i1} = \frac{\partial f_i}{\partial x_1} = \frac{u_ix_0}{u_i^2 + u_ix_2 + x_3} \\
J_{i2} = \frac{\partial f_i}{\partial x_2} = \frac{x_0(u_i^2 + u_i x_1)u_i}{(u_i^2 + u_ix_2 + x_3)^2} \\
J_{i3} = \frac{\partial f_i}{\partial x_3} = \frac{x_0(u_i^2 + u_i x_1)}{(u_i^2 + u_ix_2 + x_3)^2} \\
$$
为了到有意义的解同时避免除零问题，保证解收敛到全局最优，这里限定条件为$0 \le x_j \le 100，j=0,1,2,3$ 。下面是对x进行估算的具体实现。

```python
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
def model(x, u):
    return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])

def fun(x, u, y):
   	return model(x,u) - y

def jac(x, u, y):
    J = np.empty((u.size, x.size))
    den = u ** 2 + x[2] * u + x[3]
    num = u ** 2 + x[1] * u
    J[:,0] = num / den
    J[:,1] = x[0] * u / den
    J[:,2] = -x[0] * num * u / den ** 2
    J[:,3] = -x[0] * num / den ** 2
    return J
#拟合
u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])

x0 = np.array([2.5, 3.9, 4.15, 3.9])
res = least_squares(fun, x0, jac=jac, bounds=(0, 100), args=(u, y), verbose=1)    

u_test = np.linspace(0, 5)
y_test = model(res.x, u_test)
plt.plot(u, y, 'o', markersize=4, label='data')
plt.plot(u_test, y_test, label='fitted model')
plt.xlabel("u")
plt.ylabel("y")
plt.legend(loc='lower right')
plt.show()
```

**2. 最小二乘法拟合curve_fit**

```python
import scipy.optimize as optimize
import numpy as np
#产生数据
def f(x):
    return x**2 + 10*np.sin(x)
xdata = np.linspace(-10, 10, num=20)
ydata = f(xdata)+ np.random.randn(xdata.size)
plt.scatter(xdata, ydata, linewidths=3.0, edgecolors="red")
#plt.show()
#拟合
def f2(x,a,b):
    return a*x**2 + b*np.sin(x)
guess = [2,2]
#求得结果是使误差平方和最小的参数值
params, params_covariance = optimize.curve_fit(f2, xdata, ydata, guess)
#画出拟合的曲线
x1 = np.linspace(-10,10,256)
y1 = f2(x1,params[0],params[1])
plt.plot(x1,y1)
plt.show()
```

###3.4 自定义minimizer函数

minimize方法中的method可以是一个callable对象，因此只需要将自定义minimizer函数传递过去即可。例如，下面实现一个多元最小化方法，每一次按照固定步长搜索每一维度的邻居直到车说道最优解。

```python
from scipy.optimize import OptimizeResult
def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):
    bestx = x0
    besty = fun(x0)
    funcalls = 1
    niter = 0
    improved = True
    stop = False
    
    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        for dim in range(np.size(x0)):
            for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                testx = np.copy(bestx)
                testx[dim] = s
                testy = fun(testx, *args)
                funcalls += 1
                if testy < besty:
                    besty = testy
                    bestx = testx
                    improved = True
                    
                if callback is not None:
                    callback(bestx)
                if maxfev is not None and funcalls >= maxfev:
                    stop = True
                    break
    return OptimizeResult(fun=besty, x=bestx, nit=niter,nfev=funcalls, success=(niter > 1))

res = minimize(rosen, x0, method=custmin, options=dict(stepsize=0.05))
#Out[181]:
#     fun: 2.7093749999999135
#    nfev: 385
#     nit: 48
# success: True
#       x: array([ 1.25,  1.55,  2.45,  6.  ])
```











---

[^1]: F. Lenders, C. Kirches, A. Potschka: “trlib: A vector-free implementation of the GLTR method for iterative solution of the trust region problem”, <https://arxiv.org/abs/1611.04718>
[^2]: N. Gould, S. Lucidi, M. Roma, P. Toint: “Solving the Trust-Region Subproblem using the Lanczos Method”, SIAM J. Optim., 9(2), 504–525, (1999). <http://epubs.siam.org/doi/abs/10.1137/S1052623497322735>
[^3]: Nocedal, S.J. Wright “Numerical optimization.” 2nd edition. Springer Science (2006).
[^4]: Conn, A. R., Gould, N. I., & Toint, P. L. “Trust region methods”. Siam. (2000). pp. 169-200.