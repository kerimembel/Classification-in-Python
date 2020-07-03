import matplotlib.pyplot as plt

# Neighbor values
knn_x = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45] 
# corresponding Accuracy values 
knn_y = [0.7358722358722358,0.7908476658476659,0.7644348894348895,0.8003685503685504 ,0.7825552825552825 , 0.8058968058968059,0.7982186732186732,0.8052825552825553  ,0.7972972972972973,0.8015970515970516 ,0.8065110565110565,0.8058968058968059,0.8071253071253072 ,0.8077395577395577,0.8086609336609336,0.8071253071253072,0.8058968058968059] 

#The number of trees in the forest
rf_x = [50,100,150,200,250,300]
#Corresponding Accuracy values
rf_y = [0.8624078624078624,0.8645577395577395,0.8667076167076168,0.8639434889434889,0.8617936117936118,0.8642506142506142]

#var_smoothing in Naive Bayes
nb_x = ["1e-7","1e-8","1e-9","1e-10","1e-11","1e-12","1e-13","1e-14","1e-15","1e-16"]
#Corresponding Accuracy values
nb_y = [0.8022113022113022,0.8028255528255528,0.8028255528255528,0.8043611793611793,0.8071253071253072,0.8108108108108109,0.8117321867321867,0.8117321867321867,0.8117321867321867,0.8117321867321867]

#max_depth in Decision Tree(with Random State 0)
dt_x = [1,2,4,5,6,7,8,9,10,15,20]
#Corresponding Accuracy values
dt_y = [0.7684275184275184,0.8356879606879607,0.851965601965602,0.8571867321867321,0.8581081081081081,0.859029484029484,0.855958230958231,0.8507371007371007,0.8541154791154791,0.8528869778869779,0.8347665847665847]

#The inverse strength of the regularization 
svm_x = [1,2,3,4,5,6,10,13,15]
#Corresponding Accuracy values
svm_y = [0.8068181818181818,0.8086609336609336,0.8083538083538083,0.8068181818181818,0.8071253071253072,0.8068181818181818,0.8071253071253072,0.8071253071253072,0.8071253071253072]

# KNN Accuracy Graph
knn = plt.figure(1)
plt.plot(knn_x, knn_y) 
plt.xlabel('Neighbors') 
plt.ylabel('Accuracy') 
plt.title('KNN Accuracy Graph')  

# Random Forest Accuracy Graph
rf = plt.figure(2)
plt.plot(rf_x, rf_y) 
plt.xlabel('Tree Number') 
plt.ylabel('Accuracy') 
plt.title('Random Forest Accuracy Graph')  

#Naive Bayes Accuracy Graph
nb = plt.figure(3) 
plt.plot(nb_x, nb_y) 
plt.xlabel('Var Smoothing') 
plt.ylabel('Accuracy') 
plt.title('Naive Bayes Accuracy Graph')  

#Decision Tree Accuracy Graph
nb = plt.figure(4) 
plt.plot(dt_x, dt_y) 
plt.xlabel('Max Depth') 
plt.ylabel('Accuracy') 
plt.title('Decision Tree Accuracy Graph')  

#Support Vector Machine Accuracy Graph
nb = plt.figure(5) 
plt.plot(svm_x, svm_y) 
plt.xlabel('Regularization Value') 
plt.ylabel('Accuracy') 
plt.title('Support Vector Machine Accuracy Graph')  

plt.show() 

