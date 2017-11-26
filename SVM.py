from sklearn import svm,decomposition
import sklearn
import pandas
import numpy as np



a=pandas.read_excel('C:/Users/MiladAbedi/Documents/VT/Courses/Machine Learning/Project/ManeuverDetection_Project_Labeled_Dataset.xlsx')
a=a.drop('FILE_ID',axis=1)
a['TIME']=-a['TIME_START']+a['TIME_END']
a=a.drop('TIME_START',axis=1)
a=a.drop('TIME_END',axis=1)
a=a.drop('Diff',axis=1)
a=sklearn.utils.shuffle(a)
Data=a.drop('Labels',axis=1)
Labels=a['Labels']

train_data = Data.iloc[0:500,:]
train_label=Labels.iloc[0:500]
test_data = Data.iloc[500:]
test_label=Labels.iloc[500:]


train_data,Normalizer_Transform=normalize(train_data)
test_data=normalize(test_data,Normalizer_Transform)

pca=sklearn.decomposition.PCA()
pca.fit(test_data)
test_data=pca.transform(test_data)
train_data=pca.transform(train_data)



clf = svm.SVC(kernel='linear')
clf.fit(train_data, train_label)
print('Testing Accuracy: ',clf.score(test_data,test_label))
print('Training Accuracy:',clf.score(train_data,train_label))

Test_Con_Mat=pandas.DataFrame(sklearn.metrics.confusion_matrix(test_label,clf.predict(test_data),labels=[0,1,2,3,4,5,6,7,8]),[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8])
print('--------------------------------')
print('Testing Data Confusion matrix:')
print(Test_Con_Mat)

Train_Con_Mat=pandas.DataFrame(sklearn.metrics.confusion_matrix(train_label,clf.predict(train_data),labels=[0,1,2,3,4,5,6,7,8]),[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8])
print('--------------------------------')
print('Training Data Confusion matrix:')
print(Train_Con_Mat)
print('--------------------------------')



#print(Con_Mat)


def normalize(Input,transform=[]):
    if np.any(transform):
         Result=(Input-Input.mean())/(Input.max()-Input.min())
         return Result
    else:
        Result=(Input-Input.mean())/(Input.max()-Input.min())
        transform=np.array([Input.mean(),Input.min(),Input.max()])
        return Result,transform
    
