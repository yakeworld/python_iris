# python_iris
python iris 分类 机器学习笔记

```
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
