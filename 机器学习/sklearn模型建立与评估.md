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





