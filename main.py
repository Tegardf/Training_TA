import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold

from preprocessing import claheFilterContrast, removeFlare
from segmentation import ROI_segmentation
# , testCHT, testFrst, testContour

import pandas as pd

import pypickle


#test 2
def main2():
    categories =  ['katarak','normal']
    dataDir='Dataset/'

    x_arr = []
    y_arr = []

    for i in categories:
        path = os.path.join(dataDir,i)
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path,img_name))
            
            # print(path,img_name)
            # if img_name == '68.jpg':
            #     cv2.imshow('error img',img)
            #     cv2.waitKey(0)

            #preprocessing
            img = cv2.resize(img,(300,300),interpolation=cv2.INTER_CUBIC)
            img = removeFlare(img)
            img = claheFilterContrast(img)

            #segmentation
            # roi_pupil,_ = ROI_segmentation(img)
            roi_pupil = ROI_segmentation(img)
            roi_pupil = cv2.resize(roi_pupil,(50,50),interpolation=cv2.INTER_CUBIC)

            x_arr.append(roi_pupil.flatten())
            y_arr.append(categories.index(i))
        print(f'loaded category:{i} successfully')

    x_data = np.array(x_arr)
    y_data = np.array(y_arr)

    df = pd.DataFrame(x_data)
    df['Target']= y_data

    x = df.iloc[:,:-1] #input
    # print(x[0].shape)
    y = df.iloc[:,-1] #output 

    # param_grid = {'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
    # svc = SVC(probability=True)
    # model = GridSearchCV(svc,param_grid)
    # model = SVC(kernel='rbf',C=1.0,gamma='auto',cache_size=400,class_weight=None,coef0=0.1,decision_function_shape='ovo',degree=6, max_iter=-1,probability=True,random_state=None,shrinking=True, tol=0.001,verbose=False)

    model = SVC(kernel='rbf',C=1,gamma='scale')
    print('\n')
    scores = cross_val_score(model,x,y,cv=5,scoring='accuracy',n_jobs=16)
    print (scores)
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
    print(f"Standard Deviation of Accuracy: {std_accuracy * 100:.2f}%")

    print('\n')

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10,random_state=12,stratify=y)
    print('success split')

    model.fit(x_train,y_train)

    # print(model)

    print('The Model is trained well with the given images')

    pypickle.save('./modelSave/model3.pkl',model)

    # y_pred=model.predict(x_test)
    y_pred=model.predict(x_test)
    print('\n')

    print("The predicted Data is :")
    print(y_pred)
    print("The actual data is:")
    print(np.array(y_test))
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
    # cohen_score = cohen_kappa_score(y_test, y_pred)
    # print("Kappa Score: ",cohen_score,"\n")
    # matrix = confusion_matrix(y_test, y_pred)
    # print(matrix)

    report = classification_report(y_test,y_pred,zero_division=1)
    print("Classification Report:\n", report)
    print('\n')


    # scores = cross_val_score(model,x_train,y_train,cv=5,scoring='accuracy')
    
    

    # loo = LeaveOneOut()
    # scores = cross_val_score(model , x_test , y_test , cv = loo)
    # conf_mat = confusion_matrix(y, scores)
    # print("Confusion Matrix")
    # print(conf_mat)
    # cohen_score = (cohen_kappa_score(y, scores)*10)
    # print("Accuracy of Model LeaveOneOut is:",scores.mean() * 100)
    # print("\n")

if __name__=="__main__":
    main2()
