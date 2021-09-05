import pandas as pd
import os
import seaborn as sns
import regex as re
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import CategoricalNB
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import datasets, linear_model, metrics 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as ptl
from sklearn.feature_selection import SelectKBest #for selecting k best features
from sklearn.feature_selection import chi2#one of the feature selection technique/
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

import warnings
warnings.filterwarnings('ignore')

np.random.seed(7)

from imblearn.under_sampling import RandomUnderSampler

# Instantiate Random Under Sampler
rus = RandomUnderSampler(random_state=11)       

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
from sklearn import preprocessing
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
# label_encoder converts human readable lang to machine lang
label_encoder = preprocessing.LabelEncoder()
df = pd.read_csv(r'hpo1.csv',low_memory=False)
df1 = pd.read_csv(r'hpo1.csv',low_memory=False)
df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
df1.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df1.columns.values]
df1 = df1.drop(columns=['spam criteria'])
X = df.iloc[:,1:19].fillna(0)
y1 = df['spam criteria'].fillna('None')
y = label_encoder.fit_transform(y1)

# Perform random under sampling
df_data, df_target = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)


bestfeatures = SelectKBest(score_func = chi2,k=10)
bestfit = bestfeatures.fit(X,y)

dfscores=pd.DataFrame(bestfit.scores_)
dfcolumns=pd.DataFrame(X.columns)
features_scores = pd.concat([dfcolumns,dfscores],axis=1)
features_scores.columns=['specs','scores']

#features_scores
feat_scores=features_scores.nlargest(10,'scores')
print(feat_scores)
feature_list=list(feat_scores['scores'])# retrieve score coloumn values from "feat_scores" dataframe and convert them to list
print(feature_list)
print("<---------------------------->")
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train,y_train)



print('-------------------------For ExtaTREEClassifier-------------------------')
featureimportances = pd.Series(model.feature_importances_,index = X.columns)
featureimportances.nlargest(10).plot(kind='barh')
ptl.show()
def spam_score(y_test,pred,model_name):
    l=[]
    #features=[3.1,3.1,1.9,1.4,1.3,0.9,0.9,0.8,0.5,0.3]
    c=1
    s=0
    for i in range(len(y_test)):
        temp=(y_test[i]-pred[i])**2
        l.append(temp/c)
        c=c+1
    #print(l)
    for i in range(len(feature_list)):
        for j in range(len(l)):
            s=s+(feature_list[i]*l[j])
    s=s/c
    print(f"spam score for {model_name} :",s)
"""implementation of svc"""
 #creating a object named model2
svc_model = SVC(kernel ='rbf')
#fitting and training model"
svc_model.fit(X_train,y_train)
#prediction of svm with train set,output is a set of predicted values as a col 
svm_pred2 = svc_model.predict(X_test)
spam_score(y_test,svm_pred2,"SVC")
print(confusion_matrix(y_test,svm_pred2))
print("accuracy for svm:",metrics.accuracy_score(y_test,svm_pred2)*100)
print(metrics.classification_report(y_test,svm_pred2))
#spam_score(y_test,svm_pred2,"svm")

"""implementation of knn"""
knn_model= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn_model.fit(X_train,y_train) 
knn_y_pred= knn_model.predict(X_test) 
print(confusion_matrix(y_test,knn_y_pred))
print("accuracy for knn:",metrics.accuracy_score(y_test,knn_y_pred)*100)
print(metrics.classification_report(y_test,knn_y_pred))
spam_score(y_test,knn_y_pred,"KNN")



"""implementation of random forest"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
rfc_model=RandomForestClassifier(n_estimators=100)
rfc_model.fit(X_train,y_train)
y_pred=rfc_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("accuracy for random forest:",metrics.accuracy_score(y_test, y_pred)*100)
print(metrics.classification_report(y_test,y_pred))
spam_score(y_test,y_pred,"Random Forest")


"""implementation of decision tree"""
from sklearn.tree import DecisionTreeClassifier
dtr_model = DecisionTreeClassifier()

# Train Decision Tree Classifer
dtr_model.fit(X_train,y_train)
y_pred = dtr_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("accuracy for decision tree:",metrics.accuracy_score(y_test, y_pred)*100)
print(metrics.classification_report(y_test,y_pred))
spam_score(y_test,y_pred,"Decision tree")


"""implementation of xgboost"""
from xgboost import XGBClassifier
xg_model=XGBClassifier(eval_metric='mlogloss')
xg_model.fit(X_train,y_train)
xgpred=xg_model.predict(X_test)
print(confusion_matrix(y_test,xgpred))
print("accuracy for xgboost:",metrics.accuracy_score(y_test, xgpred)*100)
print(metrics.classification_report(y_test,xgpred))
spam_score(y_test, xgpred,"xgboost")

scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score)}

def models_evaluation(X, y, folds):
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''

    # Perform cross-validation to each machine learning classifier
    xg = cross_validate(xg_model, X, y, cv=folds,scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds,scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds,scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds,scoring=scoring)
    #knn = cross_validate(knn_model, X, y, cv=folds,scoring=scoring)

    # Create a data frame with the models performance measures scores
    models_scores_table = pd.DataFrame({'Xgboost': [xg['test_accuracy'].mean(),
                                                                xg['test_precision'].mean(),
                                                                xg['test_recall'].mean(),
                                                                xg['test_f1_score'].mean()],

                                        'Support Vector Classifier': [svc['test_accuracy'].mean(),
                                                                      svc['test_precision'].mean(),
                                                                      svc['test_recall'].mean(),
                                                                      svc['test_f1_score'].mean()],

                                        'Decision Tree': [dtr['test_accuracy'].mean(),
                                                          dtr['test_precision'].mean(),
                                                          dtr['test_recall'].mean(),
                                                          dtr['test_f1_score'].mean()],

                                        'Random Forest': [rfc['test_accuracy'].mean(),
                                                          rfc['test_precision'].mean(),
                                                          rfc['test_recall'].mean(),
                                                          rfc['test_f1_score'].mean()]},
                                        
                                       #'K-Nearest Neighbours': [knn['test_accuracy'].mean(),
                                                                      #knn['test_precision'].mean(),
                                                                      #knn['test_recall'].mean(),
                                                                      #knn['test_f1_score'].mean()],

                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Return models performance metrics scores data frame
    return models_scores_table

result_table = models_evaluation(df_data, df_target, 5)

"""Plotting Bar graph"""
ax=result_table.plot(kind="bar",figsize=(10,5))
ax.set(title="Comparison",
      xlabel="Metrics",
      ylabel="Values")
ax.legend().set_visible(True)
plt.xticks(rotation=0)
ax.set_ylim([0.0,1.0])
plt.show();
