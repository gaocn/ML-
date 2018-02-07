##统计分析包(scipy.stats)

该包提供概率统计分析中常用的分布，方法等，其中包括94个连续分布函数和13个离散分布函数。

###1. 随机变量

连续随机变量中公共的方法有

| 方法名  | 说明                                       |
| ---- | ---------------------------------------- |
| rvs  | random variates，是随机变量的一个特定输出结果，例如$stats.norm.rvs(size=3)$ 随机产生三个正态分布的值 |
| pdf  | 概率密度函数(Probability Density Function) $>> stats.norm.pdf([1,4]) \\ array([  2.41970725e-01,   1.33830226e-04])$ |
| cdf  | 累积分布函数(Cumulative Distribution Function)，是概率密度函数的积分，记$F(a)=P(x \le a) \\ >>norm.cdf([-1., 0, 1]) \\ array([ 0.15865525,  0.5,  0.84134475])$ |
| sf   | 生存函数或残存函数，表示病人、设备等对象的生存时间超过指定时间的概率。      |
| ppf  | 逆分布函数(Percent Point Function，百分点函数) $P(x \le G(a))=  a 或 x=G(a)=G(F(x)) \\ 例如查看一个分部的中位数 \  stats.norm.ppf(0.5) 结果为 0.0$ |
| isf  | 逆生存函数                                    |

**1. 修改均值和方差**

在标准分布中，随机变量X是通过$\frac{X - loc}{scale}$变换得到的，默认$loc=0，scale=1$ ，因此可以通过修改loc和scale改变标准分布函数。例如在指数分布中可以通过修改方差改变指数分布。
$$
F(x) = 1 - exp(-\lambda x)，均值为 \frac{1}{\lambda}
$$

```python
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import expon

norm.stats(loc=3, scale=4, moments="mv")
#(array(3.0), array(16.0))

expon.mean(scale=3.)
#3.0

uniform.cdf([0, 1, 2, 3, 4, 5], loc=1, scale=4)
#array([ 0.  ,  0.  ,  0.25,  0.5 ,  0.75,  1.  ])
```

 

