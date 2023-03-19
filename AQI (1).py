#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("air-quality-india.csv")
df[:5]


# In[2]:


df.info()


# In[3]:


# Missing values
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values= missing_values_table(df)
missing_values.style.background_gradient(cmap='Reds')


# In[4]:


year = df['Year'].value_counts()
print(f'Total number of Year in the dataset : {len(year)}')
print(year.index)


# In[ ]:





# In[5]:


# create a figure and axis
fig, ax = plt.subplots()

# scatter the sepal_length against the sepal_width
ax.scatter(df['PM2.5'], df['Year'])
# set a title and labels
ax.set_title('PM2.5 Level')
ax.set_xlabel('PM2.5')
ax.set_ylabel('Year')


# In[ ]:





# In[6]:


# # get columns to plot
# columns = df.columns.drop(['Day'])
# # create x data
# x_data = range(0, df.shape[0])
# # create figure and axis
# fig, ax = plt.subplots()
# # plot each column
# for column in columns:
#     ax.plot(x_data, df[column])
# # set title and legend
# ax.set_title('Level')
# ax.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


from  sklearn.model_selection import train_test_split

features = ['Year','Month','Day','Hour']

x = df.loc[:, features]
y = df.loc[:, ['PM2.5']]

X_train,X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 25)


# In[8]:


X_train_scaled = X_train.copy()
y_train_scaled = y_train.copy()
X_train_scaled['Year'] = X_train_scaled['Year'] % 100
y_train_scaled['PM2.5'] = y_train_scaled['PM2.5'] / 100

X_test_scaled = X_test.copy()
y_test_scaled = y_test.copy()
y_test_scaled['PM2.5'] = y_test_scaled['PM2.5'] / 100
X_test_scaled['Year'] = X_test_scaled['Year'] % 100



print(X_train_scaled[:10])


# In[9]:


model = keras.Sequential([
    keras.layers.Dense(200,input_shape = (4,), activation = 'tanh'),
     keras.layers.Dense(150,activation = 'relu'),
    keras.layers.Dense(1,activation = 'tanh')
])


# In[10]:


model.compile(optimizer = 'adam',
            loss = 'MAE',
             metrics = ['accuracy']
            )


# In[11]:


model.fit(X_train_scaled,y_train_scaled,epochs = 10)


# In[ ]:





# In[12]:


X_train_scaled[:5]


# 

# In[ ]:


# # import everything from tkinter module
from tkinter import * 
from tkinter import simpledialog
from tkinter import messagebox

# create a tkinter window
root = Tk()             
 
    
root.withdraw()
sum = 'Y'
while(sum!='N'):
    # the input dialog
    a = simpledialog.askstring(title="AQI",
                                      prompt="Year : ")
    b = simpledialog.askstring(title="AQI",
                                      prompt="Month : ")
    c = simpledialog.askstring(title="AQI",
                                      prompt="Day : ")
    d = simpledialog.askstring(title="AQI",
                                      prompt="Hour : ")

    def fun(num):  
        messagebox.showinfo("PM2.5 : ",n)  

    root.geometry('500x600') 


     # Set the position of button on the top of window.

    # btn.pack(side = 'bottom')     



    converted_num_a = int(a)
    converted_num_b = int(b)
    converted_num_c = int(c)
    converted_num_d = int(d)

    data = {
        'Year' : [converted_num_a%100] ,
        'Month': [converted_num_b] ,
        'Day' : [converted_num_b] ,
        'Hour' : [converted_num_b] ,
    }

    dff = pd.DataFrame(data)
    n = model.predict(dff)*100

    btn2 = Button(root, text = 'Hello', bd = '5',command = fun(n))
    btn2.pack(side = 'top')
    
    num = simpledialog.askstring(title="Test",
                                      prompt="Do you want to continue : (Y/N)")
    
    sum = num


# What is Particulate Matter 2.5 (PM2.5)? The term fine particles, or particulate matter 2.5 (PM2.5), refers to tiny particles or droplets in the air that are two and one half microns or less in width. Like inches, meters and miles, a micron is a unit of measurement for distance.

# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
 
# Read Images
img = mpimg.imread('img.png')
 
# Output Images
plt.imshow(img)


# In[ ]:





# ## 
