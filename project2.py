#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate a Medical Appointment No Shows 2016 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('/Users/xing-huanggao/Desktop/Undefined/data_analysis/unit2/project2/noshowappointments-kagglev2-may-2016.csv')


# In[3]:


# Check first few rows of this dataset
df.head()


# In[4]:


# determine the number of rows and columns
df.shape


# In[5]:


# print information about this dataframe including the index dtype and column dtypes, non-null values and memory usage.
df.info()


# In[6]:


# confirm if the dataset contains any duplicate values
df.duplicated().sum()


# In[7]:


# determine any null and empty values
df.isnull().sum()


# ####  General properties about this data set
# - Consist of 110527 rows and 14 columns
# - Zero duplicated and empty values

# ### Fix the column labels
# - Convert to lower case from upper case
# - Change the column headers
# - Validate if all these changes are completed

# In[8]:


# convert all the column head labels to lowercase
df.columns = df.columns.str.lower()


# In[9]:


# change column header
new_label = ['patient_id', 'appoint_id','gender','schedule_day','appoint_day', 'age','neighbour', 'scholarship','hypertension','diabetes','alcoholism','handcap','sms','no_show']
df.columns = new_label


# In[10]:


# confirm if the conversion from uppercase to lower case is completed
df.columns


# ### Change the data type in this data set
# - Both`scheduled_day` and `appoint_day` are object, we will change the data type to datetime by pandas ```to_datetime``` function.
# - Convert `patient_id` and `appoint_id` to object, because both of them are numeric.

# In[11]:


# convert schedulle_day and appoint_day to datetime format
df[['schedule_day','appoint_day']]= df[['schedule_day','appoint_day']].apply(pd.to_datetime)


# In[12]:


# confirm the first few rows of converted columns
df[['schedule_day','appoint_day']].head()


# In[13]:


# convert patient_id and appoint_id to object
df[['patient_id','appoint_id']] = df[['patient_id','appoint_id']].astype('object')


# ### Data Cleaning (drop off the patient with age -1 from the data set)

# ### General statistics from this data set

# In[14]:


# determin general statistics including mean, dispersion and shape of a dataset’s distribution
df.describe()


# The min age of patients is -1, thus we have to remove it prior to dig deeply in the data set.

# In[15]:


# identiy the index of -1 age patient, then drop it
df.drop(df.query('age == -1').index, inplace=True)


# In[16]:


#confirm the patient data with -1 age has been removed.
df[df.age==-1]


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 (How many paitents visited the hosipital with mutiple times ?)

# We make a pie chart with matplotlib to show the distribution between the patients with 1 time (unique) and mutiple visits.

# In[17]:


# count the number of patients with one time visit
uniq_patient = df.patient_id.nunique()


# In[18]:


# count total number of visits
mutiple_visit_patient = df.patient_id.count() - uniq_patient


# In[19]:


mutiple_visit_patient


# In[20]:


#make a pie chart with matplotlib to show the distribution between the patien
labels = ['Patients with mutiple visits', 'Unique']
sizes = [mutiple_visit_patient, uniq_patient]

# set up the size of subplot figure
fig1,ax1 = plt.subplots(figsize=(5,5))

# Use a zero explode and turn of the shadow for better visibility
ax1.pie(sizes, explode=(0,0.05), labels=labels, startangle=-45,shadow=False, radius=0.8,textprops={'size': 'larger'},autopct='%1.1f%%');
ax1.title.set_text('Distribution of patient visit frequency')


# In[21]:


# identify the patient with higest visting numbers
df.patient_id.value_counts().head(1)


# #### Conclution 1
# About 43% of total patients visited the hosiptial for mutiple times.

# ### Research Question 2  (It exists a correlation between patient with the scholarship and no-show frequence?)

# In[22]:


# group no_show data and determine the average values
df.groupby(['no_show'],as_index=False).mean()


# In[23]:


# select the patient with the scholarship, then determin the no-show frequence
scholar = df.query('scholarship ==1').no_show.value_counts(1) *100


# In[24]:


scholar


# In[25]:


# select the patient without the scholarship, then determin the no-show frequence
non_scholar = df.query('scholarship ==0').no_show.value_counts(1)*100


# In[26]:


non_scholar


# In[27]:


# creat a stacked bar plot to visulize the two groups of patients( with and without the scholarship)
head = ['Absence','Attendence']
show = (scholar['No'], non_scholar['No'])
no_show = (scholar['Yes'],non_scholar['Yes'])

# get a figure and define the figure size
fig = plt.gcf()
fig.set_size_inches(5, 5)

# define the number of bars and width of bars
ind = np.arange(2)
width = 0.2

# plot stacked bar with matplotlib pyplot function
p1 = plt.bar(ind, no_show, width, color='r');
p2 = plt.bar(ind, show, width, bottom=no_show, color='b');

# assign the label names for x, y, title and legend for the bar plot
plt.xticks(ind, ('+', '-'),fontsize=14)
plt.ylabel('Visit frequency %', fontsize=14)
plt.xlabel('Patients (scholarship)',fontsize=14)
plt.legend((p1[0],p2[0]),(head[0],head[1]),fontsize=12,loc=9, bbox_to_anchor=(0.5, 0.5))
plt.title('No-show frequency and patient with or without scholarship',fontsize=14);


# #### Conclution 2:
# - The patients without the scholarship are more likly to show up compared to the group received the scholarship.

# ### Research Question 3 (It exists a correlation between patient ages and no-show frequence?)

# In[28]:


# select the patient group with age equal or greater than 37 years from this dataset
elder_group = df.query('age >=37')

# count the number of show and no-show patients in this group
elder_show_counts = elder_group.no_show.value_counts()


# In[29]:


# sum of the total number of patients in the elder group
total_elder = elder_group.patient_id.count()


# In[30]:


# calculate the no-show and show frequences for the elder patient group
elder_show = elder_show_counts.No/total_elder*100 # show frequence
elder_no_show = 100-elder_show # no-show frequence


# In[31]:


elder_no_show


# In[32]:


# select the patient group with age equal or younger than 34.3 years from this dataset
younger_group = df.query('age <=34.3')

# count the number of show and no-show patients in the younger group
younger_show_counts = younger_group.no_show.value_counts()


# In[33]:


# sum of the total number of patients in the younger group
total_younger = younger_group.patient_id.count()


# In[34]:


# calculate the no-show and show frequences for the younger patient group
younger_show = younger_show_counts.No/total_younger*100 # show frequence
younger_no_show = 100-younger_show  # no-show frequence


# In[35]:


younger_no_show


# In[36]:


# select the patient with age between 34.3 and 37
mid_group = df.query('34.3 <age<37')

# count the number of show and no-show patients in this group
mid_show_counts = mid_group.no_show.value_counts()


# In[37]:


total_mid = mid_group.patient_id.count()
total_mid


# In[38]:


# calculate the no-show and show frequences for the younger patient group
mid_show = mid_show_counts.No/total_mid*100 # show frequence
mid_no_show = 100-mid_show  # no-show frequence


# In[39]:


mid_no_show


# In[40]:


# creat a stacked bar plot
head = ['Absence','Attendence']
no_show = (elder_no_show, mid_no_show, younger_no_show)
show = (elder_show, mid_show, younger_show )

# define the number of bars and the width of bars
ind = np.arange(3)
width = 0.3

# get a figure and define the figure size
fig = plt.gcf()
fig.set_size_inches(5, 5)

p1 = plt.bar(ind, no_show, width, color='r')
p2 = plt.bar(ind, show, width, bottom=no_show, color='b')

# assign the label names for x, y, title and legend for the bar plot
plt.xticks(ind, ('elder', 'mid','younger'),fontsize=14)
plt.ylabel('Visit frequency %', fontsize=14)
plt.xlabel('Patient age range',fontsize=14)
plt.title('No-show frequency and patient ages')
plt.legend((p1[0],p2[0]),(head[0],head[1]),fontsize=14,loc='lower left',bbox_to_anchor=(1, 0.5));


# #### conclution 3
# - The elder patients with an age greater or equal to 37 are more liky to attend their scheduled appointment as compared to mid- or younger age patients.

# ### Research Question 4 (It exists a correlation between aging and occurances of health issues?)
# - we use pair plot to visualize a relationship between aging and the occurance of diseases such as diabetes, hypertension and alcoolism from this data set.

# In[41]:


# use groupby function to split the data into groups based on ages, then determine the average values for each numeric columns.
patient = df.groupby('age',as_index=False).mean()

# convert the column 'age' to float64 prior to make a pair plot
patient['age'] = patient['age'].astype('float64')

# adjust the plot context parameters such as axis scale,linewidth
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.5})

# use seaborn pariplot function to create the scatter figure with selected columns labeled as 'age', 'hypertension','alcoholism' and 'diabetes'
scatter_matrix = sns.pairplot(patient, vars=['age', 'hypertension','alcoholism','diabetes'],diag_kind="kde", kind="reg",palette="husl", height=5) # add a linear regression fittig line with kind="reg".


# #### Conclusion 3
# 
# - a positive correlation between patient age and their diabetes occurance
# 
# - a positive correlation between patient age and their hypertension occurance
# 
# - no correlation between patient age and their alcoholism occurance

# #### Final conclusions
# 
# - About 43% of total patients visited the hosiptial for mutiple times.
# 
# - The patients without the scholarship are more likly to show up compared to the group received the scholarship.
# 
# - The elder patients with an age greater or equal to 37 are more liky to attend their scheduled appointment as compared to mid- or younger age patients.
# 
# - A positive correlation between aging and the occurance of two diseases including diabetes and hypertension.
# 
# - A positive correlation between diabetes and hypertension in patients.
# 
# #### Limitations and missing information
# 
# 
# - It will be interested to know if the occurence of diabetes and hypertension is higher in healthy people as compared to those patient data.
# 
# 

# In[ ]:




