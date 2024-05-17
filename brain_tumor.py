#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


import os
path = os.listdir('C:/Users/User/Downloads/archive (1)/Training')
classes = {'no_tumor': 0, 'pituitary_tumor': 1}


# In[3]:


import cv2
X = []
Y = []

for cls in classes:
    pth = 'C:/Users/User/Downloads/archive (1)/Training/' + cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200, 200))
        X.append(img)
        Y.append(classes[cls])


# In[4]:


X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)


# In[5]:


np.unique(Y)


# In[6]:


pd.Series(Y).value_counts()


# In[7]:


X.shape, X_updated.shape


# In[8]:


plt.imshow(X[0], cmap='gray')


# In[9]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=.20)


# In[10]:


xtrain.shape, xtest.shape


# In[11]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# # Model Training

# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[13]:


import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)  


# In[14]:


sv = SVC()
sv.fit(xtrain, ytrain)


# # Evaluation

# In[15]:


print('Training Score:', lg.score(xtrain, ytrain))
print('Testing Score:', lg.score(xtest, ytest))


# In[16]:


print('Training Score:', sv.score(xtrain, ytrain))
print('Testing Score:', sv.score(xtest, ytest))


# # Prediction

# In[17]:


pred = sv.predict(xtest)


# In[18]:


misclassified = np.where(ytest != pred)
misclassified


# In[19]:


print('Total Misclassified Samples:', len(misclassified[0]))
print(pred[36], ytest[36])


# # Testing

# In[25]:


dec = {0: 'No Tumor', 1: 'Positive Tumor'}


# In[29]:


plt.figure(figsize=(12, 8))
p = os.listdir('C:/Users/User/Downloads/archive (1)/Testing')
c = 1
for i in os.listdir('C:/Users/User/Downloads/archive (1)/Testing/no_tumor/')[:9]:
    plt.subplot(3, 3, c)
    img = cv2.imread('C:/Users/User/Downloads/archive (1)/Testing/no_tumor/'+ i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c += 1


# In[30]:


plt.figure(figsize=(12, 8))
p = os.listdir('C:/Users/User/Downloads/archive (1)/Testing')
c = 1
for i in os.listdir('C:/Users/User/Downloads/archive (1)/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4, 4, c)
    img = cv2.imread('C:/Users/User/Downloads/archive (1)/Testing/pituitary_tumor/'+ i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c += 1


# In[24]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

test_predictions = sv.predict(xtest)

conf_matrix = confusion_matrix(ytest, test_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Pituitary Tumor'], yticklabels=['No Tumor', 'Pituitary Tumor'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




