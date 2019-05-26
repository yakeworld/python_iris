# python_iris
python iris 分类 机器学习笔记
 

```python
# -*- coding: utf-8 -*-
"""
iris数据机器学习分类
isis鸢尾花数据集
iris是鸢尾植物，这里存储了其萼片和花瓣的长宽，共4个属性，鸢尾植物分三类。
该数据集一共有150个样本,包含4个特征变量，1个类别变量(划分为3类（0类、1类、2类）。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
iris = load_iris()

#print(iris.keys())
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.DESCR)

```


```python
print(iris)
```

    {'data': array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2],
           [5.4, 3.9, 1.7, 0.4],
           [4.6, 3.4, 1.4, 0.3],
           [5. , 3.4, 1.5, 0.2],
           [4.4, 2.9, 1.4, 0.2],
           [4.9, 3.1, 1.5, 0.1],
           [5.4, 3.7, 1.5, 0.2],
           [4.8, 3.4, 1.6, 0.2],
           [4.8, 3. , 1.4, 0.1],
           [4.3, 3. , 1.1, 0.1],
           [5.8, 4. , 1.2, 0.2],
           [5.7, 4.4, 1.5, 0.4],
           [5.4, 3.9, 1.3, 0.4],
           [5.1, 3.5, 1.4, 0.3],
           [5.7, 3.8, 1.7, 0.3],
           [5.1, 3.8, 1.5, 0.3],
           [5.4, 3.4, 1.7, 0.2],
           [5.1, 3.7, 1.5, 0.4],
           [4.6, 3.6, 1. , 0.2],
           [5.1, 3.3, 1.7, 0.5],
           [4.8, 3.4, 1.9, 0.2],
           [5. , 3. , 1.6, 0.2],
           [5. , 3.4, 1.6, 0.4],
           [5.2, 3.5, 1.5, 0.2],
           [5.2, 3.4, 1.4, 0.2],
           [4.7, 3.2, 1.6, 0.2],
           [4.8, 3.1, 1.6, 0.2],
           [5.4, 3.4, 1.5, 0.4],
           [5.2, 4.1, 1.5, 0.1],
           [5.5, 4.2, 1.4, 0.2],
           [4.9, 3.1, 1.5, 0.2],
           [5. , 3.2, 1.2, 0.2],
           [5.5, 3.5, 1.3, 0.2],
           [4.9, 3.6, 1.4, 0.1],
           [4.4, 3. , 1.3, 0.2],
           [5.1, 3.4, 1.5, 0.2],
           [5. , 3.5, 1.3, 0.3],
           [4.5, 2.3, 1.3, 0.3],
           [4.4, 3.2, 1.3, 0.2],
           [5. , 3.5, 1.6, 0.6],
           [5.1, 3.8, 1.9, 0.4],
           [4.8, 3. , 1.4, 0.3],
           [5.1, 3.8, 1.6, 0.2],
           [4.6, 3.2, 1.4, 0.2],
           [5.3, 3.7, 1.5, 0.2],
           [5. , 3.3, 1.4, 0.2],
           [7. , 3.2, 4.7, 1.4],
           [6.4, 3.2, 4.5, 1.5],
           [6.9, 3.1, 4.9, 1.5],
           [5.5, 2.3, 4. , 1.3],
           [6.5, 2.8, 4.6, 1.5],
           [5.7, 2.8, 4.5, 1.3],
           [6.3, 3.3, 4.7, 1.6],
           [4.9, 2.4, 3.3, 1. ],
           [6.6, 2.9, 4.6, 1.3],
           [5.2, 2.7, 3.9, 1.4],
           [5. , 2. , 3.5, 1. ],
           [5.9, 3. , 4.2, 1.5],
           [6. , 2.2, 4. , 1. ],
           [6.1, 2.9, 4.7, 1.4],
           [5.6, 2.9, 3.6, 1.3],
           [6.7, 3.1, 4.4, 1.4],
           [5.6, 3. , 4.5, 1.5],
           [5.8, 2.7, 4.1, 1. ],
           [6.2, 2.2, 4.5, 1.5],
           [5.6, 2.5, 3.9, 1.1],
           [5.9, 3.2, 4.8, 1.8],
           [6.1, 2.8, 4. , 1.3],
           [6.3, 2.5, 4.9, 1.5],
           [6.1, 2.8, 4.7, 1.2],
           [6.4, 2.9, 4.3, 1.3],
           [6.6, 3. , 4.4, 1.4],
           [6.8, 2.8, 4.8, 1.4],
           [6.7, 3. , 5. , 1.7],
           [6. , 2.9, 4.5, 1.5],
           [5.7, 2.6, 3.5, 1. ],
           [5.5, 2.4, 3.8, 1.1],
           [5.5, 2.4, 3.7, 1. ],
           [5.8, 2.7, 3.9, 1.2],
           [6. , 2.7, 5.1, 1.6],
           [5.4, 3. , 4.5, 1.5],
           [6. , 3.4, 4.5, 1.6],
           [6.7, 3.1, 4.7, 1.5],
           [6.3, 2.3, 4.4, 1.3],
           [5.6, 3. , 4.1, 1.3],
           [5.5, 2.5, 4. , 1.3],
           [5.5, 2.6, 4.4, 1.2],
           [6.1, 3. , 4.6, 1.4],
           [5.8, 2.6, 4. , 1.2],
           [5. , 2.3, 3.3, 1. ],
           [5.6, 2.7, 4.2, 1.3],
           [5.7, 3. , 4.2, 1.2],
           [5.7, 2.9, 4.2, 1.3],
           [6.2, 2.9, 4.3, 1.3],
           [5.1, 2.5, 3. , 1.1],
           [5.7, 2.8, 4.1, 1.3],
           [6.3, 3.3, 6. , 2.5],
           [5.8, 2.7, 5.1, 1.9],
           [7.1, 3. , 5.9, 2.1],
           [6.3, 2.9, 5.6, 1.8],
           [6.5, 3. , 5.8, 2.2],
           [7.6, 3. , 6.6, 2.1],
           [4.9, 2.5, 4.5, 1.7],
           [7.3, 2.9, 6.3, 1.8],
           [6.7, 2.5, 5.8, 1.8],
           [7.2, 3.6, 6.1, 2.5],
           [6.5, 3.2, 5.1, 2. ],
           [6.4, 2.7, 5.3, 1.9],
           [6.8, 3. , 5.5, 2.1],
           [5.7, 2.5, 5. , 2. ],
           [5.8, 2.8, 5.1, 2.4],
           [6.4, 3.2, 5.3, 2.3],
           [6.5, 3. , 5.5, 1.8],
           [7.7, 3.8, 6.7, 2.2],
           [7.7, 2.6, 6.9, 2.3],
           [6. , 2.2, 5. , 1.5],
           [6.9, 3.2, 5.7, 2.3],
           [5.6, 2.8, 4.9, 2. ],
           [7.7, 2.8, 6.7, 2. ],
           [6.3, 2.7, 4.9, 1.8],
           [6.7, 3.3, 5.7, 2.1],
           [7.2, 3.2, 6. , 1.8],
           [6.2, 2.8, 4.8, 1.8],
           [6.1, 3. , 4.9, 1.8],
           [6.4, 2.8, 5.6, 2.1],
           [7.2, 3. , 5.8, 1.6],
           [7.4, 2.8, 6.1, 1.9],
           [7.9, 3.8, 6.4, 2. ],
           [6.4, 2.8, 5.6, 2.2],
           [6.3, 2.8, 5.1, 1.5],
           [6.1, 2.6, 5.6, 1.4],
           [7.7, 3. , 6.1, 2.3],
           [6.3, 3.4, 5.6, 2.4],
           [6.4, 3.1, 5.5, 1.8],
           [6. , 3. , 4.8, 1.8],
           [6.9, 3.1, 5.4, 2.1],
           [6.7, 3.1, 5.6, 2.4],
           [6.9, 3.1, 5.1, 2.3],
           [5.8, 2.7, 5.1, 1.9],
           [6.8, 3.2, 5.9, 2.3],
           [6.7, 3.3, 5.7, 2.5],
           [6.7, 3. , 5.2, 2.3],
           [6.3, 2.5, 5. , 1.9],
           [6.5, 3. , 5.2, 2. ],
           [6.2, 3.4, 5.4, 2.3],
           [5.9, 3. , 5.1, 1.8]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'), 'DESCR': '.. _iris_dataset:\n\nIris plants dataset\n--------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 150 (50 in each of three classes)\n    :Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n                \n    :Summary Statistics:\n\n    ============== ==== ==== ======= ===== ====================\n                    Min  Max   Mean    SD   Class Correlation\n    ============== ==== ==== ======= ===== ====================\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\n    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)\n    ============== ==== ==== ======= ===== ====================\n\n    :Missing Attribute Values: None\n    :Class Distribution: 33.3% for each of 3 classes.\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThe famous Iris database, first used by Sir R.A. Fisher. The dataset is taken\nfrom Fisher\'s paper. Note that it\'s the same as in R, but not as in the UCI\nMachine Learning Repository, which has two wrong data points.\n\nThis is perhaps the best known database to be found in the\npattern recognition literature.  Fisher\'s paper is a classic in the field and\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\ndata set contains 3 classes of 50 instances each, where each class refers to a\ntype of iris plant.  One class is linearly separable from the other 2; the\nlatter are NOT linearly separable from each other.\n\n.. topic:: References\n\n   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to\n     Mathematical Statistics" (John Wiley, NY, 1950).\n   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\n   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System\n     Structure and Classification Rule for Recognition in Partially Exposed\n     Environments".  IEEE Transactions on Pattern Analysis and Machine\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\n   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions\n     on Information Theory, May 1972, 431-433.\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II\n     conceptual clustering system finds 3 classes in the data.\n   - Many, many more ...', 'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], 'filename': '/home/yakeworld/.julia/conda/3/lib/python3.7/site-packages/sklearn/datasets/data/iris.csv'}



```python
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)
df['Species'] = y
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#条状图显示组平均数，可以从图上看出不同的花种类中，他们的属性特点。
grouped_data=df.groupby("Species")
#用不同的花的类别分成不同的组，此数据为三组
group_mean=grouped_data.mean()
#求组平均值
group_mean.plot(kind="bar")
plt.legend(loc="upper center",bbox_to_anchor=(0.5,1.2),ncol=2)
plt.show()
#画图
```


![png](output_3_0.png)



```python
#画kde图
df.plot(kind="kde",subplots=True,figsize=(10,6))
plt.show()
```


![png](output_4_0.png)



```python
#df.plot(x='sepal length (cm)', y='sepal width (cm)', kind='scatter',c="Species")
#plt.show()

plt.scatter(df['petal length (cm)'],df['petal width (cm)'],c=df['Species'],cmap='hsv',alpha=0.5)
 
plt.show()
```


![png](output_5_0.png)



```python
df.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()
```


![png](output_6_0.png)



```python
import seaborn as sns
sns.set()           
sns.pairplot(df,vars=["sepal width (cm)", "sepal length (cm)","petal length (cm)","petal width (cm)"],hue='Species',height=4)  
plt.show()
```


![png](output_7_0.png)



```python
#为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(X) 
X_std = std.transform(X)
#进行数据分割，测试数据占20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y,test_size=0.4, random_state= 0)
```


```python
"""
感知机（perceptron）是二类分类的线性分类模型，其输入为实例的特征向量，输出为实例的类别。
感知机对应于输入空间（特征空间）中将实例划分为正负两类的分离超平面。
"""
from sklearn.linear_model import Perceptron
clf= Perceptron(eta0=0.1)
clf.fit(X_train, y_train)
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))
y_test_pred = clf.predict(X_test) 
xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()    

plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
```

    train accuracy: 0.900
    test accuracy: 0.850



![png](output_9_1.png)



```python

#https://github.com/seven0525/Kaggle/blob/0800179136c04dd9f0dc920c7ca9ee9bc70f0749/models/classifications/.ipynb_checkpoints/%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-checkpoint.ipynb
param_grid = {'eta0': [0.001, 0.01, 0.1, 0.5, 1]}
from sklearn.model_selection import StratifiedKFold
kf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
clf = GridSearchCV(Perceptron(), param_grid, cv=kf_5)
clf.fit(X_train, y_train)
print("best-parameters:{}".format(clf.best_params_))
print("mean-score:{}".format(clf.best_score_))
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))

y_test_pred = clf.predict(X_test)
xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
```

    best-parameters:{'eta0': 0.01}
    mean-score:0.9111111111111111
    train accuracy: 0.900
    test accuracy: 0.850


    /home/yakeworld/.julia/conda/3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



![png](output_10_2.png)



```python
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1000.0)
clf.fit(X_train,y_train)
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))
#print('accuracy:%.2f'%metrics.accuracy_score(y_test, y_test_pred))
#print('confusion matirx:\n', metrics.confusion_matrix(y_test, y_test_pred))
y_test_pred = clf.predict(X_test)
xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
```

    train accuracy: 1.000
    test accuracy: 0.967



![png](output_11_1.png)



```python
"""
在scikit-learn中，与逻辑回归有关的主要是这3个类。
LogisticRegression， LogisticRegressionCV 和logistic_regression_path。
其中LogisticRegression和LogisticRegressionCV的主要区别是LogisticRegressionCV使用了交叉验证来选择正则化系数C。
而LogisticRegression需要自己每次指定一个正则化系数。除了交叉验证，以及选择正则化系数C以外，
LogisticRegression和LogisticRegressionCV的使用方法基本相同。

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y,test_size=0.4, random_state= 0)
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=3,solver='lbfgs',multi_class='multinomial')
#solver：‘newton-cg’,'lbfgs','liblinear','sag'  default:liblinear
#'sag'=mini-batch
#'multi_clss':
clf.fit(X_train, y_train)
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))
#print('accuracy:%.2f'%metrics.accuracy_score(y_test, y_test_pred))
#from pylab import mpl

y_test_pred = clf.predict(X_test)
plt.rcParams['font.sans-serif']=['AR PL UKai CN']
plt.rcParams['axes.unicode_minus']=False
x_test_len = range(len(X_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(0.5,3.5)
plt.plot(x_test_len, y_test, 'ro',markersize = 6, 
  zorder=3, label=u'真实值')
plt.plot(x_test_len, y_test_pred, 'go', markersize = 10, zorder=2,
         label=u'Logis算法预测值,$R^2$=%.3f' % model.score(X_test, y_test))
plt.legend(loc = 'lower right')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'鸢尾花数据分类', fontsize=20)
plt.show()

xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()

plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
```

    train accuracy: 0.978
    test accuracy: 0.933



![png](output_12_1.png)



![png](output_12_2.png)



```python
# 设置字符集，防止中文乱码
"""
from matplotlib.font_manager import FontManager
import subprocess
 
fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)
print (mat_fonts)
output = subprocess.check_output('fc-list :lang=zh -f "%{family}\n"', shell=True)
print ('*' * 10, '系统可用的中文字体', '*' * 10)
print (output)
zh_fonts = set(f.split(',', 1)[0] for f in output.decode().split('\n'))
available = mat_fonts & zh_fonts
print ('*' * 10, '可用的字体', '*' * 10)
for f in available:
    print(f)

"""
```




    '\nfrom matplotlib.font_manager import FontManager\nimport subprocess\n \nfm = FontManager()\nmat_fonts = set(f.name for f in fm.ttflist)\nprint (mat_fonts)\noutput = subprocess.check_output(\'fc-list :lang=zh -f "%{family}\n"\', shell=True)\nprint (\'*\' * 10, \'系统可用的中文字体\', \'*\' * 10)\nprint (output)\nzh_fonts = set(f.split(\',\', 1)[0] for f in output.decode().split(\'\n\'))\navailable = mat_fonts & zh_fonts\nprint (\'*\' * 10, \'可用的字体\', \'*\' * 10)\nfor f in available:\n    print(f)\n\n'




```python
"""
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

"""
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y,test_size=0.4, random_state= 0)
from sklearn.tree import DecisionTreeClassifier  
clf = DecisionTreeClassifier()  
clf.fit(X_train, y_train)  
#print(clf)   
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))
y_test_pred = clf.predict(X_test)  
#print(y_test_pred)
#from sklearn import metrics
#print('Accuracy:%.2f'%metrics.accuracy_score(y_test, y_test_pred))
xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
dot_data=tree.export_graphviz(clf)
graph = graphviz.Source(dot_data) 
graph.render("tree")

```

    train accuracy: 1.000
    test accuracy: 0.950



![png](output_14_1.png)





    'tree.pdf'




```python
#K近邻法(k-nearest neighbors,KNN)是一种很基本的机器学习方法了，在我们平常的生活中也会不自主的应用。
"""
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30,
p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)[source]
n_neighbors： 选择最邻近点的数目k
weights： 邻近点的计算权重值，uniform代表各个点权重值相等
algorithm： 寻找最邻近点使用的算法
leaf_size： 传递给BallTree或kTree的叶子大小，这会影响构造和查询的速度，以及存储树所需的内存。
p： Minkowski度量的指数参数。p = 1 代表使用曼哈顿距离 (l1)，p = 2 代表使用欧几里得距离(l2)，
metric： 距离度量，点之间距离的计算方法。
metric_params： 额外的关键字度量函数。
n_jobs： 为邻近点搜索运行的并行作业数。
"""
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
#acc = knn.score(X_test,y_test)
#print('Accuracy:%.2f'%acc)
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))
y_test_pred = clf.predict(X_test)
#print('Accuracy:%.2f'%metrics.accuracy_score(y_test, y_test_pred))

xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
```

    train accuracy: 1.000
    test accuracy: 0.933



![png](output_15_1.png)



```python
#用sklearn的朴素贝叶斯 训练数据集
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
#acc = gnb.score(X_test,y_test)
#print('Accuracy:%.2f'%acc)
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))

y_test_pred = clf.predict(X_test)
xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()

plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


 
```

    train accuracy: 0.978
    test accuracy: 0.933



![png](output_16_1.png)



```python
"""
支持向量机（Support Vector Machine, SVM）是一类按监督学习（supervised learning）方式对数据进行二元分类（binary classification）
的广义线性分类器（generalized linear classifier），其决策边界是对学习样本求解的最大边距超平面（maximum-margin hyperplane）
"""
from sklearn import svm
clf=svm.SVC()
clf.fit(X_train,y_train)
#acc = clf.score(X_test,y_test)
#print('Accuracy:%.2f'%acc)
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))

y_test_pred = clf.predict(X_test)
xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

```

    /home/yakeworld/.julia/conda/3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    train accuracy: 0.978
    test accuracy: 0.933



![png](output_17_2.png)



```python
"""
激活	{‘identity’，‘logistic’，‘tanh’，‘relu’}，默认’relu’ 隐藏层的激活函数：‘identity’，无操作激活，对实现线性瓶颈很有用，返回f（x）= x；‘logistic’，logistic sigmoid函数，返回f（x）= 1 /（1 + exp（-x））；‘tanh’，双曲tan函数，返回f（x）= tanh（x）；‘relu’，整流后的线性单位函数，返回f（x）= max（0，x）
slover	{‘lbfgs’，‘sgd’，‘adam’}，默认’adam’。权重优化的求解器：'lbfgs’是准牛顿方法族的优化器；'sgd’指的是随机梯度下降。'adam’是指由Kingma，Diederik和Jimmy Ba提出的基于随机梯度的优化器。注意：默认解算器“adam”在相对较大的数据集（包含数千个训练样本或更多）方面在训练时间和验证分数方面都能很好地工作。但是，对于小型数据集，“lbfgs”可以更快地收敛并且表现更好。

"""
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(10),activation='relu',solver='lbfgs')
clf.fit(X_train,y_train)
#acc = clf.score(X_test,y_test)
#print('Accuracy:%.2f'%acc)
print('train accuracy: %.3f' % clf.score(X_train, y_train))
print('test accuracy: %.3f' % clf.score(X_test, y_test))

y_test_pred = clf.predict(X_test)
xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

```

    train accuracy: 1.000
    test accuracy: 0.967



![png](output_18_1.png)



```python


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

iris = load_iris()
#为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(X) 
X_std = std.transform(X)
#进行数据分割，测试数据占20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y,test_size=0.4, random_state= 0)


ytrain = np_utils.to_categorical(y_train) 


clf = Sequential()
clf.add(Dense(output_dim=10, input_dim=4))
#clf.add(Activation("relu"))
clf.add(Activation('sigmoid'))
clf.add(Dense(output_dim=3))
clf.add(Activation("softmax"))
#clf.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
#clf.fit(X_train, ytrain, nb_epoch=500, batch_size=120)
clf.fit(X_train, ytrain, nb_epoch=100, batch_size=1, verbose=0)

#clf.add(Dense(16, input_shape=(4,)))
#clf.add(Dense(output_dim=10, input_dim=4))
#clf.add(Activation('sigmoid'))
#clf.add(Dense(3))
#clf(Activation('softmax'))
#clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
#clf.fit(X_train, ytrain, nb_epoch=100, batch_size=1, verbose=0)



#print('train accuracy: %.3f' % clf.score(X_train, y_train))
#print('test accuracy: %.3f' % clf.score(X_test, y_test))
y_test_pred  = clf.predict_classes(X_test, batch_size=1)

print('Accuracy:%.2f'%metrics.accuracy_score(y_test, y_test_pred))



xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


```

    /home/yakeworld/.julia/conda/3/lib/python3.7/site-packages/ipykernel_launcher.py:22: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(input_dim=4, units=10)`
    /home/yakeworld/.julia/conda/3/lib/python3.7/site-packages/ipykernel_launcher.py:25: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(units=3)`
    /home/yakeworld/.julia/conda/3/lib/python3.7/site-packages/ipykernel_launcher.py:30: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.


    Accuracy:0.93



![png](output_19_2.png)



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(1234)

iris = load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state= 0)



#hyperparameters
hl = 10
lr = 0.01
num_epoch = 500

#build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
net = Net()

#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#train
for epoch in range(num_epoch):
    X1 = Variable(torch.Tensor(X_train).float())
    Y1 = Variable(torch.Tensor(y_train).long())

    #feedforward - backprop
    optimizer.zero_grad()
    out = net(X1)
    loss = criterion(out, Y1)
    loss.backward()
    optimizer.step()

#get prediction
X2 = Variable(torch.Tensor(X_test).float())
Y2 = torch.Tensor(y_test).long()
out = net(X2)
y_test_predicted = torch.max(out.data, 1)
#get accuration
print('Accuracy:%.2f'%metrics.accuracy_score(y_test, y_test_pred))

xf = pd.DataFrame(X_test, columns = iris.feature_names)
xf['Species'] = y_test
xf['pred']=y_test_pred
xf.head()
#xf.plot(x='petal length (cm)', y='petal width (cm)',kind='scatter',c="Species")
#xf.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter',c="pred")
#plt.show()
plt.title(u'鸢尾花分类对比', fontsize=20)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


```

    Accuracy:0.95



![png](output_20_1.png)



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import svm
#from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier



iris = load_iris()
#为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(X) 
X_std = std.transform(X)
#进行数据分割，测试数据占20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y,test_size=0.4, random_state= 0)




models={
	"MLPClassifier":MLPClassifier(hidden_layer_sizes=(10),solver='lbfgs',learning_rate_init=0.01,max_iter=500),#神经网络
    "LogisticRegression": LogisticRegression(solver='lbfgs', multi_class='multinomial',C=1000.0),
    "DecisionTreeClassifier":DecisionTreeClassifier(),#决策树分析
    "svm":svm.SVC(),
    "GaussianNB":GaussianNB(),
 
    "LogisticRegressionCV":LogisticRegressionCV(Cs=np.logspace(-4,1,50), cv=3,fit_intercept=True, penalty='l2', solver='lbfgs',tol=0.01, multi_class='multinomial'),
    "KNeighborsClassifier":KNeighborsClassifier(n_neighbors=5,weights='uniform', algorithm='auto',  p=2, metric='minkowski', n_jobs=1),
    "Perceptron":Perceptron(eta0=0.1)
   # "KMeans":KMeans()
    
}

for name,clf in models.items():
    clf.fit(X_train, y_train)
    #print(name+ 'train accuracy: %.3f' % clf.score(X_train, y_train))
    print(name+ 'test accuracy: %.3f' % clf.score(X_test, y_test))
    #score=cross_validation.cross_val_score(model,x,y,cv=5).mean()
    #print(name,score)
for name,clf in models.items():
    plt.title(name+u'鸢尾花分类对比', fontsize=20)
    plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['Species'],cmap='hsv',alpha=0.5)
    plt.scatter(xf['petal length (cm)'],xf['petal width (cm)'],c=xf['pred'],cmap='viridis',alpha=0.5)
    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    plt.show()
```

    MLPClassifiertest accuracy: 0.967
    LogisticRegressiontest accuracy: 0.967
    DecisionTreeClassifiertest accuracy: 0.950
    svmtest accuracy: 0.933
    GaussianNBtest accuracy: 0.933


    /home/yakeworld/.julia/conda/3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    LogisticRegressionCVtest accuracy: 0.933
    KNeighborsClassifiertest accuracy: 0.933
    Perceptrontest accuracy: 0.850



![png](output_21_3.png)



![png](output_21_4.png)



![png](output_21_5.png)



![png](output_21_6.png)



![png](output_21_7.png)



![png](output_21_8.png)



![png](output_21_9.png)



![png](output_21_10.png)


