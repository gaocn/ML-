## 主题模型(LDA, Latent Dirichlet Allocation)



LDA的应用方向

- 信息提取和搜索，可用于语义分析
- 文档分类/聚类、文章摘要、社区挖掘
- 基于内容的图像聚类、目标识别及其他计算机视觉应用
- 生物信息数据的应用

朴素贝叶斯可以胜任许多文本分类问题，但无法解决语料中**一词多义**和**多词一义**的问题，可以把朴素贝叶斯看做是**词法分析**，而非**语义分析**。如果使用词向量作为文档的特征，一词多义和多词一义会造成计算文档间相似度的不准确性，为此，可以通过增加“主题”的方式，一定程度的解决这样的问题：

1. **一词多义**，一个词可能被映射到多个主题中，例如“大理”在小说中可能被映射为“武侠小说”，在旅游上下文中可能被映射为“旅游胜地”；
2. **多词一义**，多个词可能被映射到某个主题的概率很高。

**$\Gamma$函数**

![Gamma函数](imgs_md/Gamma函数.png)
$$
\Gamma(x) = (x - 1) \cdot \Gamma(x - 1)  \Rightarrow \frac{\Gamma(x)}{\Gamma(x - 1) }= x - 1
$$
$\Gamma$函数是阶乘在实数上的推广，$\Gamma(x) = \int_{0}^{+\infin} t^{x-1}e^{-t}dt = (x-1)!$

**Beta分布**













###参考文献

[^1]: Blei D M, Ng A Y, Jordan M I. Latent dirichlet allocation[J]. Journal of machine Learning research, 2003, 3(Jan): 993-1022. 
[^2]: Hoffman M, Bach F R, Blei D M. Online learning for latent dirichlet allocation[C]//advances in neural information processing systems. 2010: 856-864. 
[^3]: Mihalcea R, Tarau P. Textrank: Bringing order into text[C]//Proceedings of the 2004 conference on empirical methods in natural language processing. 2004. 
[^4]: [Parameter estimation for text analysis - arbylon](http://www.arbylon.net/publications/text-est2.pdf)









