#Import Pandas for dataframe operations 
import pandas as pd

#Import classification algorithms from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#Import train and test split function and cross val score for n-fold cross validation
from sklearn.model_selection import train_test_split,cross_val_score

#Import LabelEncoder to transform columns
from sklearn.preprocessing import LabelEncoder  

#Import metrics to evaluate and analyze outputs 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Import timeit to calculate execution time of algorithms
import timeit


#Read data into dataframe
data = pd.read_csv('data\\adult.data',)

#Remove row with ' ?' values
#data = data[(data != ' ?').all(axis=1)]

#Encode columns where a column type is an object
for column in data.columns:
    if data[column].dtype == type(object):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

#Choose label or target column
y = data.iloc[:,14]
#Others data columns
X = data.iloc[:,:14]

#Test size ratio is 10% as stated in the report. 
#Random state parameter is a random seed.
#I use it to try certain results again.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=19)

#Running algorithms 

print("_____________________________________________________")  
start = timeit.default_timer()
DTC = DecisionTreeClassifier(max_depth = 8,splitter = "best", random_state=0)
DTC.fit(X_train,y_train)
DTC_prediction = DTC.predict(X_test)
accuracies = cross_val_score(DTC, X_test, y_test, cv=10)
stop = timeit.default_timer()
print("Decision Tree")
print("Accuracy : " ,accuracy_score(DTC_prediction,y_test))
print("N-fold cross validation score : ", accuracies.mean())
print(classification_report(DTC_prediction,y_test))
print('Execution Time: ', stop - start)  
print("_____________________________________________________")  

start = timeit.default_timer()
NB = GaussianNB(var_smoothing=1e-13);
NB.fit(X_train,y_train)
NB_prediction = NB.predict(X_test)
accuracies = cross_val_score(NB, X_test, y_test, cv=10)
stop = timeit.default_timer()
print("\nNaive Bayes")
print("Accuracy : " ,accuracy_score(NB_prediction,y_test))
print("N-fold cross validation score : ", accuracies.mean())
print(classification_report(NB_prediction, y_test))
print('Execution Time: ', stop - start)  
print("_____________________________________________________")  

start = timeit.default_timer()
KNN = KNeighborsClassifier(n_neighbors=35)
KNN.fit(X_train,y_train)
KNN_prediction = KNN.predict(X_test)
accuracies = cross_val_score(KNN, X_test, y_test, cv=10)
stop = timeit.default_timer()
print("\nK-Nearest Neighbors")
print("Accuracy : " ,accuracy_score(KNN_prediction, y_test))
print("N-fold cross validation score : ", accuracies.mean())
print(classification_report(KNN_prediction, y_test))
print('Execution Time: ', stop - start)  
print("_____________________________________________________")  

start = timeit.default_timer()
RF=RandomForestClassifier(n_estimators=300)
RF.fit(X_train,y_train)
RF_prediction = RF.predict(X_test)
accuracies = cross_val_score(RF, X_test, y_test, cv=10)
stop = timeit.default_timer()
print("\nRandom Forest Classifier")
print("Accuracy : " ,accuracy_score(RF_prediction,y_test))
print("N-fold cross validation score : ", accuracies.mean())
print(classification_report(RF_prediction,y_test))
print('Execution Time: ', stop - start)  
print("_____________________________________________________")  


start = timeit.default_timer()
SVM = SVC(C=2)
SVM.fit(X_train,y_train)
SVM_prediction = SVM.predict(X_test)
accuracies = cross_val_score(SVM, X_test, y_test, cv=10)
stop = timeit.default_timer()
print("\nSupport Vector Machine")
print("Accuracy : " ,accuracy_score(SVM_prediction, y_test))
print("N-fold cross validation score : ", accuracies.mean())
print(classification_report(SVM_prediction,y_test))
print('Execution Time: ', stop - start)  
print("_____________________________________________________")  

start = timeit.default_timer()
MLP_NN = MLPClassifier(hidden_layer_sizes=(14,14,14), activation='identity', max_iter=300)
MLP_NN.fit(X_train,y_train)
MLP_NN_predictiction = MLP_NN.predict(X_test)
#accuracies = cross_val_score(MLP_NN, X_test, y_test, cv=10)
stop = timeit.default_timer()
print("\nNeural Network")
print("Accuracy : " ,accuracy_score(MLP_NN_predictiction,y_test))
print("N-fold cross validation score : ", accuracies.mean())
print(classification_report(MLP_NN_predictiction,y_test))
print('Execution Time: ', stop - start)
print("_____________________________________________________")  
