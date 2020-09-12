#!/usr/bin/env python
# coding: utf-8

# TASK 2 (DATA SCIENCE AND ANALYTICS INTERNSHIP)

# # AIM: To Explore Supervised Machine Learning
# 

# # Detailed Explaination:
# In this Regression Task, an attempt is made to predict the percentage of marks a student is expected to score based upon the number of hours they have put in studying.This is a "simple Linear Regression" task.

# # Importing all the neccessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the Dataset remotely

# In[2]:


url_dataset = "http://bit.ly/w-data"
student_data = pd.read_csv(url_dataset)
#printing the Confirmation that data is read
print("Dataset read successfully")


# # Printing the First 12 lines of the dataset

# In[3]:


student_data.head(12)


# # Printing the Last 12 lines of the dataset

# In[4]:


student_data.tail(12)


# # Printing the information our dataset consists of :

# In[5]:


student_data.info()


# # The concise description:

# In[6]:


student_data.describe()


# # Total number Rows and Columns:

# In[7]:


student_data.shape


# There are 25 Rows(each student data) and 2 Columns(hours and Scores)

# Let's plot the data points of the following dataset on 2-D Graph to get a clear picture and check if we can manaully make some predictions about the relationship between the two Variables. 

# # Plotting a 2-D Graph

# In[15]:


#plotting the distribution of scores with respect to hours studied
student_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours v/s Percentage')
plt.xlabel('Hours of Studying')
plt.ylabel('Score in Percentage')
plt.grid()
plt.show()


# # Observation 
# there is a positive linear relation between the number of hours studied and percentage of score.

# # Dividing the data

# In[20]:


#The next step is to divide the data into inputs and outputs i.e; attributes and Labels
X = student_data.iloc[:, :-1].values  
y = student_data.iloc[:, 1].values 


# # Splitting the data into training and test sets 

# In[21]:


#importing the built-in method train_test_split 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0) 


# # Training the Simple Regression Model 

# In[22]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training the model is successful.")


# # Predicting the results

# In[26]:


#printing the testing data
print(X_test)


# In[24]:


# predicting the results
y_pred = regressor.predict(X_test)
#print the prediction of scores
print(y_pred)


# # Plotting the Regression Line

# In[25]:


line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # Comparing the Actual with the Predicted

# In[27]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df) 


# # Predicted Score if the Students study for 9.25 hours

# In[30]:


X_new = np.array([9.25])
X_new = X_new.reshape(-1,1)
Y_Pred = regressor.predict(X_new)
print('Number of Hours Studied is {}'.format(X_new))
print('The predicted score if studied for the said time in a day is {}'.format(Y_Pred))


# # Evaluating the Model

# In[31]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# # --------------------------------------------END OF TASK-----------------------------------------------

# In[ ]:




