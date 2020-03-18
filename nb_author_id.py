#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""


#import os 

#os.system('sudo pip install scikit-learn')  
import sys
import numpy as np
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#k= KFold(n=len(data))
#for features_train, features_test 
#features_train=((features_train).reshape(-1,1))
labels_train=(np.array(labels_train))
#labels_train = np.argmax(labels_train, axis=1)
#features_test=((features_test).reshape(-1,1))
labels_test=(np.array(labels_test))
    #labels_test = np.argmax(labels_test, axis=1)
features_train
print(labels_train)
print(features_train.shape)
print(labels_train.shape)
print(features_test.shape)
print(labels_test.shape)
    #########################################################
    ### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
t1 = time()
y_pred=clf.predict(features_test)
print ("training time:", round(time()-t1, 3), "s")
    #features_train=features_train.reshape(-1,1)
    #y_pred = clf.fit((np.matrix(features_train)), (np.matrix(labels_train))).predict(np.transpose(np.matrix(labels_test)))
accuracy= accuracy_score(y_pred, labels_test)
print ( accuracy )

#########################################################


