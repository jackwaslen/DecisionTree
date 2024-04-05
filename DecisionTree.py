#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import math

def load_data():
    # countVectorizer to extract features to text
    # each headline can be represented by numerical feature vector
    fakeArray = []
    realArray = []
    with open("/Users/base/Desktop/fake.txt", "r") as file:
        fakeArray = [line.strip() for line in file.readlines()]
        
    with open("/Users/base/Desktop/real.txt", "r") as file:
        realArray = [line.strip() for line in file.readlines()]
        
    data = fakeArray + realArray
    labels = [0] * len(fakeArray) + [1] * len(realArray)

    vectorizer = CountVectorizer()
    result = vectorizer.fit_transform(data)
    df = pd.DataFrame(data=result.toarray(), columns = vectorizer.get_feature_names_out())
    
    x_training_validation, x_testing, y_training_validation, y_testing = train_test_split(df, labels,  test_size=0.15, random_state=35)
    x_training, x_validation, y_training, y_validation = train_test_split(x_training_validation, y_training_validation, test_size=0.17647, random_state=35)
    return x_training, x_validation, y_training, y_validation, vectorizer.get_feature_names_out()


# In[ ]:





# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


def select_model():
    plotArray = []
    bestAccuracy = -9999
    bestHype = -999
    bestDepth = 0
    for i in range(3,9):
        Dtree = DecisionTreeClassifier(max_depth=i, criterion="entropy")
        resultFit = Dtree.fit(x_training, y_training)
        predictions = resultFit.predict(x_validation)
        accuracy = accuracy_score(y_validation, predictions)
        plotArray.append(accuracy)
        print("Max Depth: "+str(i)+" Accuracy: "+str(accuracy))
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestDepth = i
            bestHype = Dtree.get_params()
            
    print("Best Accuracy: "+str(bestAccuracy))
    print("Best HyperParameters: "+str(bestHype))
    plt.plot(range(3,9), plotArray)
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracies")
    plt.title("Validation Accuracy vs. max_depth")
    
    Besttree = DecisionTreeClassifier(max_depth=i, criterion="entropy")
    Bestfit = Besttree.fit(x_training, y_training)
    figure = plt.figure(figsize=(25,25))
    tree.plot_tree(Besttree, feature_names= featureNames, class_names={0: "Fake", 1: "Real"}, max_depth = 2)


# In[3]:


x_training, x_validation, y_training, y_validation, featureNames = load_data()

select_model()


# In[ ]:




