

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


# In[23]:


df = pd.read_csv(r"C:\Users\Dell\Desktop\qb64\house_prices.csv")


# In[24]:


df.head()


# In[25]:


df.fillna(df.median(numeric_only=True), inplace=True)


# In[26]:


X = df[['Home','Price','SqFt', 'Bedrooms', 'Bathrooms', 'Offers','Brick','Neighborhood']]


# In[27]:


y = df['Price'].copy()
X.fillna(0) 
y.fillna(y.median())


# In[48]:


df.isnull().sum()
df = pd.get_dummies(df, drop_first=True)
df


# In[52]:


x = df.drop(['Price'], axis = 1)
y = df['Price']


# In[53]:


import sklearn
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xtrain, ytrain)


# In[54]:


ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)


# In[55]:


df=pd.DataFrame(ytrain_pred,ytrain)
df=pd.DataFrame(ytest_pred,ytest)


# In[56]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(ytest, ytest_pred)
print(mse)
mse = mean_squared_error(ytrain_pred,ytrain)
mse


# In[57]:


import matplotlib.pyplot as plt
import numpy as np

plt.scatter(ytrain ,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest,ytest_pred ,c='lightgreen',marker='s',label='Test data')
plt.xlabel('True values')
plt.ylabel('Predicted')
plt.title("True value vs Predicted value")
plt.legend(loc= 'upper left')
plt.plot()
plt.show() 


# In[ ]:




