#Sklearn建模与评估

##一、简单线性回归

```python
import pandas as pd 
import matplotlib.pyplot as plt 

columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year','origin', 'car name' ]
cars = pd.read_table(r'G:\data\auto-mpg.data', names=columns, delim_whitespace=True)
# print(cars)

# 查看特征与结果的关系
# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)
# cars.plot("weight", "mpg", kind="scatter", ax=ax1)
# cars.plot("acceleration", "mpg",kind="scatter", ax=ax2)
# plt.show()

# 使用scikit-learn库建立回归模型
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#
#输入：X为N*M的矩阵，N为样本个数n_samples，M为特征个数m_features
#输入：Y为样本标签值，单标签（预测结果只有一个值，为列向量）和双标签（预测出来的结果不止一个，为矩阵）
lr.fit(cars[['weight']], cars['mpg'])
# lr.fit(cars['weight'], cars['mpg']) 
pred = lr.predict(cars[['weight']])

# plt.scatter(cars['weight'], cars['mpg'], c='r')
# plt.plot(cars['weight'], pred, c='b')
# plt.show()

# 衡量模型的好坏：
#     均方误差MSE，Mean Squared Error = 1/n sum[(y' - y)**2] 
#     预测值与真实值之间的均方误差
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(pred, cars['mpg'])
sqrt_mse = mse ** (.5)
print('MSE: %f \nsqrt(MSE): %f'  % (mse, sqrt_mse))
```

##二、简单逻辑回归

```python
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

admissions = pd.read_csv(r'G:\data\admissions.csv', na_filter=True)
print(admissions.head())
# plt.scatter(admissions['gpa'], admissions['admit'])
# plt.show()

#逻辑回归
# sigmoid(x) = 1 / (1 + exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(x))

# sigmod函数图
x = np.linspace(-6, 6, 50, dtype=np.float)
y = sigmoid(x)
# plt.plot(x, y)
# plt.show()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(admissions[['gpa']], admissions['admit'])
pred_prob = model.predict_proba(admissions[['gpa']])
fitted_labels = model.predict(admissions[['gpa']])
# plt.scatter(admissions['gpa'], pred_prob[:, 1])
# plt.show()

admissions['pred_labels'] = fitted_labels
print(admissions)

#不能仅仅通过精度评判模型？实际上精度不靠谱
admissions['actual_labels'] = admissions['admit']
matched = admissions['actual_labels'] == admissions['pred_labels']
correct_pred = admissions[matched]
precision = len(correct_pred) / float(len(admissions))
print('precision: %f' % precision)
```

| Prediction  | Observation           |                       |
| ----------- | --------------------- | --------------------- |
|             | Admitted(1)           | Rejected(0)           |
| Admitted(1) | True Positive (TP)，  | False Positive (FP)， |
| Rejected(0) | False Negative (FN)， | True Negative (TN)，  |

假如某个班级有男生80人，女生20人，共计100人。目标是找出所有女生。现在某人挑出50人，其中20人是女生，另外还错误的把30个男生也当作女生挑选出来了。

|            | 相关，正类                           | 无关，负类                           |
| ---------- | ------------------------------------ | ------------------------------------ |
| 被检索到   | TP，正类判定为正类，确实是女生       | FP，负类判定为正类，男生被判定为女生 |
| 未被检索到 | FN，正类判定为负类，女生被判定为男生 | TN，负类判定为负类，男生被判定为男生 |

通过上述表格，可以清楚的得到：TP=20，FP=30，FN=0，TN=50

**True Positive Rate(Sensitivity)**

TPR指标衡量模型检测正例的效果，例如用模型检测病人是否患癌症，TP表示患病的人被正确的检测出来了，而FN则是患病的人被认为是正常的，这时候结果就严重的，在这个问题上需要考虑TPR，否则好多人会因为这个模型而受难。
$$
TPR =  \frac{True  \ Positives}{True \  Positives + False \  Nagetives}
$$
**True Negative Rate**

TNR指标衡量模型检测负例例的效果，例如用模型检测病人是否患癌症，TP是没患癌症的人被正确的检测出来了，TN是没患病的人被检测出来患有癌症。
$$
TNR = \frac{True \ Negative}{False \ Positive + True \ Negative}
$$
为什么说精度经常是不准确的、有欺骗性？例如有100个样本，其中有90个样本是属于1这个类别，而10个样本属于0这个类别，让分类器预测，假设模型全部预测为1类别，则精度也有90%，因此精度在样本非常不平衡的条件下是非常不准确的。很多情况下需要衡量TPR、TNR指标。

```python
true_positive_filter = (admissions['pred_labels'] == 1) & (admissions['actual_labels'] == 1)
true_positives = len(admissions[true_positive_filter])
false_negative_filter = (admissions['pred_labels'] == 0) & (admissions['actual_labels'] == 1)
false_negatives = len(admissions[false_negative_filter])

sensitivity = true_positives / (true_positives + false_negatives)

# 过拟合：在训练数据表现很好，在测试数据表现很差！
# ROC曲线，综合查看fpr，tpr的值
from sklearn import metrics
pred_probs = model.predict_proba(admissions[['gpa']])
# threshold：
fpr, tpr, thresholds = metrics.roc_curve(admissions['actual_labels'], pred_probs)
# 画出ROC曲线
# plt.plot(fpr, tpr)
# plt.show()

# 希望ROC的横轴和纵轴都趋近于1。ROC曲线表现模型预测正例和预测负例的综合效果
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(test['actual_labels'], tests_probs[:, 1])
# 得分值越接近于1，模型效果越好。
print(auc_score)
```

##三、交叉验证

样本数据是要划分为训练集、测试集，那么存在一种情况导致切分后数据分布不均衡，例如在测试集中存在异常点比较多或训练集中异常点比较多，这就导致在训练模型是得到的结果不会很准确。如下图所示，将样本集切缝成大小相同的五个部分，如各自有100个数据，取其中的一份作为测试集，其余的4份作为训练集，每一次迭代的训练集和测试集不同，通过交叉的方式训练模型，有训练得到的5个模型共同得出最终的模型（如求5个模型结果的平均值作为最终结果），这种训练模型的方式称为交叉验证。

![叉验证图](imgs_md/交叉验证图例.png)

```python
# 交叉验证，数据集的拆分
# 取一部分作为训练数据，一部分作为测试数据
# 过拟合：在训练数据表现很好，在测试数据表现很差！
shuffled_index = np.random.permutation(admissions.index)
shuffled_admissions = admissions.loc[shuffled_index]
admissions = shuffled_admissions.reset_index()
admissions.ix[0:128,   "fold"] = 1
admissions.ix[129:257, "fold"] = 2
admissions.ix[257:386, "fold"] = 3
admissions.ix[387:514, "fold"] = 4
admissions.ix[515:644, "fold"] = 5
admissions['fold'] = admissions['fold'].astype('int')
```

**Sklearn库做交叉验证**

sklearn.model_selection.KFold(n_folds=3, shuffle=False, random_state=None)

- n_folds：需要将观测集切分成几份
- shuffle：是否随机排列观测集
- random_state：shuffle=True时，指定随机排列观测集的random seed

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

admissions = pd.read_csv(r'G:\data\admissions.csv', na_filter=True)
admissions['actual_labels'] = admissions['admit']
admissions = admissions.drop('admit', axis=1)
kf = KFold(5, shuffle=True, random_state=8)
lr = LogisticRegression()

accurates = cross_val_score(lr, admissions[['gpa']], admissions['actual_labels'], scoring='roc_auc', cv=kf)
# array([0.70175439, 0.5023511 , 0.60750507, 0.59722222, 0.67074074])
average_accuracy = sum(accurates) / len(accurates)
#  0.6159147034200947
```

















