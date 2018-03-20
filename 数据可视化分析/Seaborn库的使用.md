##[Seaborn](https://seaborn.pydata.org/tutorial)库的使用

Seaborn其实是在Matplotlib的基础上进行了更高级的API封装（模板，只需提供数据既能够展现），从而使得作图更加容易，在大多数情况下使用Seaborn就能做出很具有吸引力的图。能理解pandas的DataFrame类型，所以它们一起可以很好地工作。



**整体布局设置：seaborn.set**

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import numpy as np

# 使用Seaborn默认的画图参数，可以设置
sb.set()
```



**Seaborn的五种主题风格**

```python
# darkgrid、whitegrid、dark、white、ticks

sb.set_style("whitegrid")
data = np.random.normal(size=(20,6)) + np.arange(6) / 2.
sb.boxplot(data=data)
plt.show()
```





###热度图

```python
# darkgrid、whitegrid、dark、white、ticks

sb.set_style("whitegrid")
data = np.random.normal(size=(20,6)) + np.arange(6) / 2.
sb.heatmap(data)
plt.show()
```

