## 第二章 线性神经网络



与单层感知器结构类似，不同的是**增加支持$purelin: y=x$激活函数**，除了二值输出外还可以支持模拟输出，因此除了充当分类器还可以实现类似回归的效果。线性神经网络支持除了分类、回归外，使用多个输出/分离平面（Madaline网络）可以变相解决线性不可分问题，同时可以引入非线性成分（升维，类似SVM）一定程度上解决非线性问题。

![线性神经网络](../%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/imgs_md/%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.png)

线性神经网络采用**Delta学习规则**即梯度下降法的一般性学习规则，目标函数是**最小化平方误差（MSE）**，学习规则为权值变化量(Delta)正比于负梯度，比例系数为学习率。



## Delta学习规则

人工神经元的作用是对于输入向量$x=(x_1,x_2,...,x_n)$产生一个输出![y](https://www.zhihu.com/equation?tex=y)。为了让神经元能够给出我们期望的输出，需要训练它，或者说让它学习到一个模型。训练的样本是一系列已知的![\boldsymbol{x}](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7Bx%7D)和![\hat{y}](https://www.zhihu.com/equation?tex=%5Chat%7By%7D)，我们用![\hat{y}](https://www.zhihu.com/equation?tex=%5Chat%7By%7D)表示期待得到的正确输出。 





阅读作业
1 精读《Understanding the difficulty of training deep feedforward neural networks》一文


书面作业
1 根据《Understanding the difficulty of training deep feedforward neural networks》回答下列问题
1）normalized的初始化权值和standard的初始化权值在定义上有何不同？
2）根据文中图11，比较一下几种激活函数的优劣，什么原因造成sigmoid效果比较差？而softsign效果相对较好？
2 用任何一种编程语言编程实现LMS学习算法，并将运行测试的结果抓图







##参考文献

[^1]: Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks[C]//Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010: 249-256. 