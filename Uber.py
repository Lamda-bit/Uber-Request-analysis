#!/usr/bin/env python
# coding: utf-8

# # Project Title: Uber Request data analysis
# 

# **Analyzing Uber Request Data to Optimize Service Reliability and Customer Satisfaction:**
# Uber, as a leading ride-hailing service, collects vast amounts of data regarding ride requests, including information about the date, time, location, driver details, and customer behavior. However, challenges persist in maintaining service reliability and ensuring customer satisfaction due to various factors such as ride cancellations, driver availability, and customer experience.
# 

# ### Project Objective
# 
# To Gather, preprocess, analyze and perform Exploratory Data Analysis (EDA) on the Uber request data to identify patterns, trends, ride cancellations,driver alloction predict demand and key factors affecting service reliability, driver availability, and customer satisfaction.

# In[76]:

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder


def main():
    st.header("Uber Request data analysis")
    expander = st.expander("Data Frames", expanded=False)
    with expander:
        st.subheader(body="Initial dataset")
        firsthead
        #st.text("")
        st.text('Expanding the dataset according to timestamp')
        expandedset
        st.text("")
        st.text("Checking dimension of dataset")
        shape
        st.text("Checking columns and data type")
        info
        st.text("Checking size of dataset")
        size
        st.text("")
        st.subheader("INSIGHTS")
        
        insColumns = st.columns((2,2))
        
        with insColumns[0]:     
            st.text('Checking for the maximum Day-name rides were completed accross all days of the week')
            maxdnRidecomp
            maxdnRidecomp1
            st.text("Checking NaN value count")
            null
            st.text("Checking for the maximum hour rides were requrested accross all days of the week")
            rHH
            rHH.loc[rHH['count'] == max(rHH['count'])]
            
               
        with insColumns[1]:
            st.text("Checking for the maximum day of the week rides were completed accross all days of the week")
            rddn
            rddn.loc[rddn['count'] == max(rddn['count'])]
            st.text("Checking for the maximum frequency of ride status")
            rSs
            rSs.loc[rSs['count'] == max(rSs['count'])]
            st.text("Checking for the maximum day of the week rides were requested accross all days of the week")
            rdd
            rdd.loc[rdd['count'] == max(rdd['count'])]
        
       

    page = st.sidebar.selectbox(
        "Univariate Frequency Plots",
        [
            "Status",
            "Driver Count",
            "Day Request Frequency",
            "Drop Day Frequency",
            "Request Hour Frequency",
			"Drop Hour Frequency",
			"Day-Night Request Frequency",
			"Day-Night Drop Frequency",
			"Pick-Up Point Frequency", 
        ]
    )
    page2 = st.sidebar.selectbox(
        "Bivariate Plots",
        [
            "Weekday vs Pickup Point",
            "Request Hour vs Drop Hour",
            "Weekday vs Day",
            
        ]
    )
    page3 = st.sidebar.selectbox(
        "Multivariational Plots",
        [
            "Request Hour vs Weekay vs Drop Hour",
            "Weekday vs Day vs Drop Day-Night Period",
            "Weekday vs Day vs Request Day-Night Period",
            "Request vs Drop Hour vs Day",
            
        ]
    ) 
    page4 = st.sidebar.selectbox(
        "Correlation and Machine Learning",
        [
			"Correlation",
            "Scatter Plot of Drivers",
			"Cluster",
            "Scaled Cluster",
            "Sum of Squared Error",
            
        ]
    ) 
    columns = st.columns((2,2))
    
    with columns[0]:
	
        if page == "Status":
            status()
        elif page == "Driver Count":
            countDriver()
        elif page == "Day Request Frequency":
            request_day()
        elif page == "Drop Day Frequency":
            dropdayFrequency()
        elif page == "Request Hour Frequency":
            requestHour()
        elif page == "Drop Hour Frequency":
            dropHour()
        elif page == "Day-Night Request Frequency":
            dayNightrequests()
        elif page == "Day-Night Drop Frequency":
            dayNightdrop()
        elif page == "Pick-Up Point Frequency":
            pickupPoint()
            
    with columns[1]:
             
        if page2 == "Weekday vs Pickup Point":
           weekdayPickup()
        elif page2 == "Request Hour vs Drop Hour":
           requestvsDrophour()
        elif page2 == "Weekday vs Day":
           weekDayvsday()
    
    with columns[0]:
   
        if page3 == "Request Hour vs Weekay vs Drop Hour":
            Request_hour_vs_Weekday_v_Drop_hour()
        elif page3 == "Weekday vs Day vs Drop Day-Night Period":
            Week_Day_v_Day_v_dropdaynight()
        elif page3 == "Weekday vs Day vs Request Day-Night Period":
            Week_Day_v_Day_v_reqdaynight()
        elif page3 == "Request vs Drop Hour vs Day":
            Request_v_Drop_hour_v_Dayofthe_month()
            
    with columns[1]:	
        if page4 == "Correlation":
            heatmap()
        elif page4 == "Cluster":
            cluster()
        elif page4 == "Scatter Plot of Drivers":
            scatter()
        elif page4 == "Scaled Cluster":
            scaledPlot()
        elif page4 == "Sum of Squared Error":
            elbow()
    st.markdown("___")

    ml_expander = st.expander("Machine Learning Frames", expanded=False)
    with ml_expander:
       ml()
       
	
   
# # Data Examination

# In[77]:

df = pd.read_csv("UberRequestData.csv", parse_dates=["Request_timestamp"])
firsthead = df
df['Request_timestamp'] = pd.to_datetime(df['Request_timestamp'], format='mixed', dayfirst=True)
df['Drop_timestamp'] = pd.to_datetime(df['Drop_timestamp'], format='mixed', dayfirst=True)

#Fillins NaN Values with zero (0)
df = df.fillna(0)
#First 10 row index

df.head(10)


# In[78]:


#from bottom up
df.tail()


# # Data Cleaning
# 
# * Splitting the Request_timestamp and Drop_timestamp into, Date, Hour, Day, Day of the week and weekday

# In[79]:


#Adding date request hour, drop hour, week day column to the dataset

df['date'] = pd.DatetimeIndex(df['Request_timestamp']).date
df['Request_Hour'] = pd.DatetimeIndex(df['Request_timestamp']).hour
df['Drop_Hour'] = pd.DatetimeIndex(df['Drop_timestamp']).hour
df['Week_Day'] = pd.DatetimeIndex(df['Request_timestamp']).weekday
df['Day'] = pd.DatetimeIndex(df['Request_timestamp']).day
df['Day_Name'] = pd.DatetimeIndex(df['Request_timestamp']).day_name()
df['DropDay_Name'] = pd.DatetimeIndex(df['Drop_timestamp']).day_name()

#changing into categories of day and night
df['Request_day-night'] = pd.cut(x=df['Request_Hour'],
							bins = [0,11,15,19,24],
							labels = ['Morning','Afternoon','Evening','Night'])

df['Drop_day-night'] = pd.cut(x=df['Drop_Hour'],
							bins = [0,11,15,19,24],
							labels = ['Morning','Afternoon','Evening','Night'])							
expandedset = df
#Outputting first 12 row index 
df.head(12)


# In[80]:


#Checking columns and data type
df.info()
info = df.info()

# ### Examination Output
# 
# - The dataset was gotten from kaggle.
# - There are 6745 rows and 15 columns.

# In[81]:


#Checking dimension of dataset
shape = df.shape
#---df.shape


# In[82]:


#Checking size of dataset
size = df.size
#---df.size


# In[83]:

null =df.isna().sum()
df.isna().sum()


# - There is a total of 4119 null values (Drop_day-night=4020, Request_day-night=99).

# In[84]:


# Checking NaN value count
df.isna().values.sum()


# In[85]:


# Checking Null/Nan value count
df.isnull().values.sum()


# In[86]:


# Summary Statistics of datset
df.describe()


# In[87]:


df.head()


# In[88]:


df.tail()


# ## INSIGHTS

# ### Maximum and Minimum ride request hour

# In[89]:


#Checking for the maximum hour rides were requrested accross all days of the week
rH = df['Request_Hour']
rHH = pd.DataFrame(rH.value_counts())
rHH.sort_values(['Request_Hour'], ascending=True)

rHH = rHH.reset_index()

#---rHH


# In[90]:


#---rHH.loc[rHH['count'] == max(rHH['count'])]


# ### Maximum and Minimum drop time of day-night period

# In[91]:


#Checking for the maximum Day-name rides were completed accross all days of the week
rDH = df['Drop_day-night']
rdh = pd.DataFrame(rDH.value_counts())
rdh.sort_values(['Drop_day-night'], ascending=False)

rdh = rdh.reset_index()
maxdnRidecomp = rdh
#---rdh


# In[92]:

maxdnRidecomp1 =rdh.loc[rdh['count'] == max(rdh['count'])]
#---rdh.loc[rdh['count'] == max(rdh['count'])]


# In[93]:


#Checking for the maximum day-night range rides were requrested accross all days of the week
rDN = df['Request_day-night']
rdn = pd.DataFrame(rDN.value_counts())
rdn.sort_values(['Request_day-night'], ascending=False)
rdn = rdn.reset_index()
#---rdn


# In[94]:


#---rdn.loc[rdn['count'] == max(rdn['count'])]


# ### Maximum and Minimum weekday ride requests

# In[95]:


#Checking for the maximum day of the week rides were requested accross all days of the week
rDD = df['Day_Name']
rdd = pd.DataFrame(rDD.value_counts())
rdd.sort_values(['Day_Name'], ascending=False)
rdd = rdd.reset_index()
#---rdd


# In[96]:


#---rdd.loc[rdd['count'] == max(rdd['count'])]


# ### Maximum and Minimum ride completed weekday

# In[97]:


#Checking for the maximum day of the week rides were completed accross all days of the week
Group = df.groupby(df.Status)
comp =Group.get_group("Trip Completed")
rDdn = comp['DropDay_Name']
rddn = pd.DataFrame(rDdn.value_counts())
rddn.sort_values(['DropDay_Name'], ascending=False)
rddn = rddn.reset_index()
#---rddn


# In[98]:


#---rddn.loc[rddn['count'] == max(rddn['count'])]


# ### Ride Status Frequency

# In[99]:


#Checking for the maximum frequency of ride status 
rS = df['Status']
rSs = pd.DataFrame(rS.value_counts())
rSs.sort_values(['Status'], ascending=True)

rSs = rSs.reset_index()
#---rSs


# In[100]:


#---rSs.loc[rSs['count'] == max(rSs['count'])]


# In[101]:


#Data Visualisation
df.head()


# ### Maximum and Minimum Driver frequency

# In[102]:


#Checking for the maximum frequency of ride status of drivers
did = df[df.Driver_id>0]
did = did.Driver_id
DID = pd.DataFrame(did.value_counts())
DID.sort_values(['Driver_id'], ascending=False)

DID = DID.reset_index()
DID.head()


# In[103]:


#---DID.loc[DID['count'] == max(DID['count'])]


# ## Data visualisation

# ### Univariate Analysis

# In[104]:


#Visualising driver counts

def countDriver():
    st.subheader("Visualising driver counts")
    driverFrame = df[df.Driver_id>0]
    fig =plt.figure(figsize=(10,60))
    a = sns.countplot(data = driverFrame , y = driverFrame.Driver_id )
    a.bar_label(a.containers[0])
    plt.title("Frequency of drivers who worked in July", color = 'Brown', weight = 'bold').set_fontsize(16)
    plt.ylabel('Driver ID')
    st.pyplot(fig)
    st.caption("Based on the chart, Driver id 27 has the maximum ride count of 22.")


#  #### Based on the chart, Driver id 27 has the maximum ride count of 22.

# In[105]:


df.head()


# **How many rides were cancelled, completed and unavailable ?**

# In[106]:

    
def status():
	st.subheader("Plotting the status of rides requested by users")
	fig = plt.figure(figsize=(10,3))
	b = sns.countplot(y = df.Status, palette ='crest')
	b.bar_label(b.containers[0])
	plt.xlabel('Count', color ="#176B87")
	plt.ylabel('Status', color ="#176B87")
	plt.title('Counts of ride status', weight ='bold', fontdict={'fontsize': 20}, color ="#176B87")
	st.pyplot(fig)
	st.caption("The above visual shows that more trips were completed but the second highest has no cars availabe so more cars will have to be deployed")

# **Which weekday has a lot of rides and which has less ?**

# In[107]:


#Plotting request days of rides
def request_day():
    st.subheader("Plotting request days of rides")
    fig = plt.figure(figsize=(8,6))
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"]
    c = sns.countplot(data = df, x = df.Day_Name,  palette ='magma', order = order)
    c.bar_label(c.containers[0])
    plt.xlabel('WEEK DAYS', color ="#9A4444" )
    plt.ylabel('Count',color = "#9A4444")
    plt.title('Ride request day frequency', weight ='bold', fontdict={'fontsize': 20}, color ="#9A4444")
    st.pyplot(fig)
    st.caption("From the chart, Saturday has the highest rides and monday has least. No work on sundays")


# From the chart, Saturday has the highest rides and monday has least. No work on sundays

# In[108]:


#Plotting the days of the week rides were completed
Group = df.groupby(df.Status)
comp =Group.get_group("Trip Completed")


# #### Frequency of Days on which rides gets completed

# In[109]:

def dropdayFrequency():
    st.subheader("Plotting days on which rides were completed")
    fig = plt.figure(figsize=(8,4))
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"]
    d = sns.countplot(data = comp, y = comp.DropDay_Name, palette= sns.color_palette("plasma"), order = order)
    d.bar_label(d.containers[0])
    plt.ylabel('WEEK DAYS', color ="#5B0888" )
    plt.xlabel('Count',color = "#5B0888")
    plt.title('Drop day frequency', weight ='bold', fontdict={'fontsize': 20}, color ="#5B0888")
    st.pyplot(fig)
    st.caption("Thursday appears to top this chart. This also includes rides orded in previous day completed in thursday")

# Thursday appears to top this chart. This also includes rides orded in previous day completed in thursday

# **Which hour has the maximum and minimum request ?**

# In[110]:


#Plotting the Hours rides were requested
def requestHour():
    st.subheader("Plotting the Hours rides were requested")
    fig = plt.figure(figsize=(10,7))
    e = sns.countplot(data = df, y = df.Request_Hour, palette= sns.color_palette("rocket"))
    e.bar_label(e.containers[0])
    plt.ylabel('Request Hour', color ="#662549" )
    plt.xlabel('Count',color = "#662549")
    plt.title('Request Hour Chart', weight ='bold', fontdict={'fontsize': 20}, color ="#662549")
    st.pyplot(fig)
    st.caption("From the chart it appears 18th hour has the maximum request of 510 counts")

# From the chart it appears 18th hour has the maximum request of 510 counts

# **Which hour has the maximum and minimum completed period ?**

# In[111]:


#Plotting the Hours rides were completed
Group = df.groupby(df.Status)
completed =Group.get_group("Trip Completed")
DH = completed['Drop_Hour']
dh = pd.DataFrame(DH.value_counts())
dh.sort_values(['Drop_Hour'])
dh = dh.reset_index()
#---dh.loc[dh['count']==max(dh['count'])]


# In[112]:

def dropHour():
    st.subheader("Plotting frequency of Ride completed hour")
    fig = plt.figure(figsize=(10,7))
    e = sns.countplot(data = completed, y = completed.Drop_Hour, palette= sns.color_palette("inferno"))
    e.bar_label(e.containers[0])
    plt.ylabel('Drop Hour', color ="#662549" )
    plt.xlabel('Count',color = "#662549")
    plt.title('Drop Hour Chart', weight ='bold', fontdict={'fontsize': 20}, color ="#662549")
    st.pyplot(fig)
    st.caption("From the chart it appears the 6th hour has the maximum drop hour of 190 counts")

# From the chart it appears the 6th hour has the maximum drop hour of 190 counts

# In[113]:


df.head()


# **Which time of the day has the highest and lowest rides ?**

# In[114]:


#Plotting time of the day rides were requested
def dayNightrequests():
    st.subheader("Plotting time of the day rides were requested")
    fig = plt.figure(figsize=(10,3))
    f = sns.countplot(data = df, y = df['Request_day-night'], palette= sns.color_palette("inferno"))
    f.bar_label(f.containers[0])
    plt.ylabel('Day - Night', color ="#662549" )
    plt.xlabel('Count',color = "#662549")
    plt.title('Day-Night requests period', weight ='bold', fontdict={'fontsize': 20}, color ="#662549")
    st.pyplot(fig)
    st.caption("Morning has the highest request period based on the chart")

# Morning has the highest request period based on the chart

# In[115]:


#Plotting time of the day rides were completed
Group = df.groupby(df.Status)
completed =Group.get_group("Trip Completed")
DH = completed['Drop_day-night']
dh = pd.DataFrame(DH.value_counts())
dh.sort_values(['Drop_day-night'])
dh = dh.reset_index()
#---dh.loc[dh['count']==max(dh['count'])]


# In[116]:


n = df['Drop_day-night']
n.value_counts()


# **Which time of the day has the highest rides completed ?**

# In[117]:

def dayNightdrop():
    st.subheader("Plotting Day-Night period of rides completed")
    fig =plt.figure(figsize=(10,3))
    g = sns.countplot(data = completed, y = completed['Drop_day-night'], palette=['#432371','gold'])  
    g.bar_label(g.containers[0])
    plt.ylabel('Day - Night', color ="#662549" )
    plt.xlabel('Count',color = "#662549")
    plt.title('Day-Night drop period', weight ='bold', fontdict={'fontsize': 20}, color ="#662549")
    st.pyplot(fig)
    st.caption("It appears morning has the highest ride completed followed by Night time")

# It appears morning has the highest ride completed followed by Night time

# **Which has highest frequency between airport and city**

# In[118]:

def pickupPoint():
#Plotting city vs airport frequency under pickup point
    st.subheader("Plotting city vs airport frequency under pickup point")
    fig = plt.figure(figsize=(10.5,2))
    f = sns.countplot(data = df, y = df['Pickup_point'], palette=['#432371','gold'])
    f.bar_label(f.containers[0])
    plt.ylabel('Pickup Point', color ="#662549" )
    plt.xlabel('Count',color = "#662549")
    plt.title('Pick up Point frequency', weight ='bold', fontdict={'fontsize': 20}, color ="#432371")
    st.pyplot(fig)
    st.caption("Rides with the city has the highest pickups compared to the proximity of the airport but since the difference is not too huge this is not much problem")

# Rides with the city has the highest pickups compared to the proximity of the airport but since the difference is not too huge this is not much problem

# In[119]:


df.head()


# ## Bivariate Analysis
# 

# **Which weekday has the highest pick up ?**

# In[120]:


#Bivariate
def weekdayPickup():
    st.subheader("Plot of weekday against Pick-up point")
    fig = plt.figure(figsize=(8,5))
	#Plot of weekday against Pick-up point
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"]
    h =sns.countplot(x='Day_Name', data=df, hue='Pickup_point', order=order, palette=['#432371','gold'])
    plt.title('Weekday by Pickup Point', weight='bold', color = "#432371").set_fontsize('16')
    h.bar_label(h.containers[0])
    h.bar_label(h.containers[1])
    plt.xlabel('DAYS OF WEEK', color = "#432371")
    plt.ylabel('Count', color = "#432371")
    st.pyplot(fig)
    st.caption("Saturday turn to top this chart but equals on requests within the city")

# ### Request hour vs Drop hour

# In[121]:

def requestvsDrophour():
    st.subheader("Plotting Request and drop hours")
	#Plotting Request and drop hours
    fig = plt.figure(figsize=(8,5))
    I =sns.scatterplot(x='Drop_Hour', data=df, y='Request_Hour', color = "brown")
    plt.title('Request and Drop hour', weight='bold', color = "Brown").set_fontsize('16')
    plt.xlabel('Drop Hour', color = 'Brown')
    plt.ylabel('Request Hour', color = 'Brown')
    st.pyplot(fig)

# ### Week day vs Day of the month

# In[122]:

def weekDayvsday():
    st.subheader("Week day vs Day of the month")
    fig, ax = plt.subplots()
    I =sns.scatterplot(x='Week_Day', data=df, y='Day', color = "brown")
    plt.title('Week Day and Day', weight='bold', color = 'Brown').set_fontsize('16')
    plt.xlabel('Week Day', color = 'Brown')
    plt.ylabel('Day', color = 'Brown')
    st.pyplot(fig)

# ## Multivariate Analysis

# In[123]:


#Multivariational
df.head()


# ## **What is the relationship between request hour, drop hour and the day of the week?**

# In[124]:


#multivariate analysis cells:" request hour v weekday v drop hour"
def Request_hour_vs_Weekday_v_Drop_hour():
    st.subheader("Request hour v Weekday v Drop hour")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Request_Hour', y='Drop_Hour',hue='Day_Name',data=df , palette='deep')
    plt.title('Request hour vs Weekday v Drop hour', weight='bold', color = 'Brown').set_fontsize('16')
    plt.xlabel('Request Hour', color = 'Brown')
    plt.ylabel('Drop Hour', color = 'Brown')
    st.pyplot(fig)


# In[125]:

def Week_Day_v_Day_v_dropdaynight():
    st.subheader("Plotting WeekDay vs Day vs Drop-Day")
    fig, ax = plt.subplots()
    I =sns.scatterplot(x='Week_Day', data=df, y='Day', hue = 'Drop_day-night', palette ="rocket", color = "brown")
    plt.title('Week Day and Day', weight='bold', color = 'Brown').set_fontsize('16')
    plt.xlabel('Week Day', color = 'Brown')
    plt.ylabel('Day', color = 'Brown')
    st.pyplot(fig)


# In[126]:

def Week_Day_v_Day_v_reqdaynight():
    st.subheader("Plotting WeekDay vs Day vs Day-Night Request")
    fig, ax = plt.subplots()
    I =sns.scatterplot(x='Week_Day', data=df, y='Day', hue = 'Request_day-night', palette ="rocket", color = "brown")
    plt.title('Week Day and Day', weight='bold', color = 'Brown').set_fontsize('16')
    plt.xlabel('Week Day', color = 'Brown')
    plt.ylabel('Day', color = 'Brown')
    st.pyplot(fig)


# In[127]:


#Plotting Request and drop hours
def Request_v_Drop_hour_v_Dayofthe_month():
    st.subheader("Plotting Request Vs Drop hour Vs Day")
    fig = plt.figure(figsize=(8,5))
    I =sns.scatterplot(x='Drop_Hour', data=df, y='Request_Hour',hue='Day', palette ="flare", color = "brown")
    plt.title('Request and Drop hour, and Day of the month', weight='bold', color = "Brown").set_fontsize('16')
    plt.xlabel('Drop Hour', color = 'Brown')
    plt.ylabel('Request Hour', color = 'Brown')
    st.pyplot(fig)

# In[128]:


df.head(3)


# ### Correlation

# In[129]:


#Correlation
dfn = df
dfn = dfn.drop(['Request_timestamp','Drop_timestamp','date', 'Request_id','Driver_id', 'Week_Day',	'Day','Day_Name',	'DropDay_Name'	], axis = 1)
object_cols = ['Pickup_point', 'Status','Request_day-night','Drop_day-night']
OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dfn[object_cols]))
OH_cols.index = dfn.index
OH_cols.columns = OH_encoder.get_feature_names_out()
dfn_final = dfn.drop(object_cols, axis=1)
dfn = pd.concat([dfn_final, OH_cols], axis=1)
dfn.head()


# In[130]:


dfn.corr()


# **Correlation plot between Request Hour, Drop Hour, Pickup Point, ride status and day-night request and drop periods**

# In[131]:

def heatmap():
    st.markdown("Correlation plot between Request Hour, Drop Hour, Pickup Point, ride status and day-night request and drop periods")
    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(dfn.corr(), 
    			cmap='RdYlBu', 
    			fmt='.2f', 
    			linewidths=2, 
    			annot=True)
    st.pyplot(fig)
    st.caption(" The higher the positive number the stronger the positive correlation (deeper blue)" 
    			+ "The higher the negative number the Stronger the negative correlation (deeper brown)"
    			+ "The more it approaches zero (0) the less or no correlation.(white blend)")

# The higher the positive number the stronger the positive correlation (deeper blue)
# The higher the negative number the Stronger the negative correlation (deeper brown)
# The more it approaches zero (0) the less or no correlation.(white blend)

# In[132]:


df.head()


# ### Exporting Driver count for machine learning to check clustering of drivers

# In[133]:


#DrF =pd.DataFrame(driverFrame)
driverFrame = df[df.Driver_id>0]
DrF = driverFrame.Driver_id.value_counts()
DrF = pd.DataFrame(DrF)
DrF.sort_values(by = 'Driver_id', axis=0,  inplace=True)
DrF.reset_index()
DrF.to_csv('Driver_count.csv')


#-------------------------------------------------------------------------------------

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import streamlit as st


def ml():
    columns = st.columns((2,1))
    
    with columns[0]:
        st.header("Dataset of Driver count")
        mdf
        st.subheader("Groupng drivers into 5 clusters with KMeans")
        km
        st.subheader("New dataset with predicted cluster column")
        CL
        st.subheader("Cluster Frames")
        clusterframes()
        st.subheader("Getting the Sum of Squared Errors SSE")
        sse
       


    with columns[1]:
        st.subheader("Scaled Frames")
        scaled
        scaledFrames()
        
        st.subheader("Applying the Gausian model below")
        st.subheader("Testing sample cluster values")
        testing_y
        st.subheader("Model Predicted")
        predictM
        st.subheader("Score of the model")
        score
        st.caption("The score above is very low hence low accuracy and inconsistency in counts based on clusters")
        st.text(" ")
    







# In[2]:


mdf = pd.read_csv('Driver_count.csv')

#mdf.head()


# ## Scatter plot of drivers 

# In[21]:

def scatter():
    st.subheader("Scatter plot of drivers")
    fig = plt.figure(figsize=(14,5))
    sns.scatterplot(data = mdf , x =mdf.Driver_id, y = mdf['count'])
    st.pyplot(fig)


# ### Groupng drivers into 5 clusters

# In[4]:


km = KMeans(n_clusters=5)


# In[5]:


y_predict = km.fit_predict(mdf[['Driver_id','count']])



# In[6]:


mdf['cluster'] = y_predict
CL = mdf
mdf.head()


# In[7]:




mdf1 = mdf[mdf.cluster == 0]
mdf2 = mdf[mdf.cluster == 1]
mdf3 = mdf[mdf.cluster == 2]
mdf4 = mdf[mdf.cluster == 3]
mdf5 = mdf[mdf.cluster == 4]

def clusterframes():
    st.markdown('Frame with digit 0 encododer')
    mdf1 
    st.markdown('Frame with digit 1 encododer')
    mdf2 
    st.markdown('Frame with digit 2 encododer')
    mdf3 
    st.markdown('Frame with digit 3 encododer')
    mdf4 
    st.markdown('Frame with digit 4 encododer')
    mdf5 

def cluster():
    st.subheader("5 Cluster of Drivers")
    fig = plt.figure(figsize=(14,5))

    sns.scatterplot(data=mdf1, x=mdf1.Driver_id, y = mdf1['count'], label = 'mdf1')
    sns.scatterplot(data=mdf2, x=mdf2.Driver_id, y = mdf2['count'], label = 'mdf2')
    sns.scatterplot(data=mdf3, x=mdf3.Driver_id, y = mdf3['count'], label = 'mdf3')
    sns.scatterplot(data=mdf4, x=mdf4.Driver_id, y = mdf4['count'], label = 'mdf4')
    sns.scatterplot(data=mdf5, x=mdf5.Driver_id, y = mdf5['count'], label = 'mdf5')

    plt.xlabel('Driver ID')
    plt.ylabel('count')
    plt.title('Cluster plot of Drivers', weight ='bold', color = 'Brown').set_fontsize(16)
    st.pyplot(fig)


# ### Scaling the clusters from 0 to 1 for more accuracy of clusters

# In[8]:


scaler = MinMaxScaler()
scaled = scaler.fit_transform(mdf[['Driver_id','count']].to_numpy())
scaled = pd.DataFrame(scaled , columns=['Driver_id','count'])
scaled.head()
y_predict = km.fit_predict(mdf[['Driver_id','count']])
scaled['cluster'] = y_predict
#scaled


# In[9]:


centers = km.cluster_centers_
c = scaler.fit_transform(centers)


# In[10]:


scaled1 = scaled[scaled.cluster == 0]
scaled2 = scaled[scaled.cluster == 1]
scaled3 = scaled[scaled.cluster == 2]
scaled4 = scaled[scaled.cluster == 3]
scaled5 = scaled[scaled.cluster == 4]

def scaledFrames():
    st.text("")
    st.markdown("Scaled with digit 0")
    scaled1
    st.markdown("Scaled with digit 1")
    scaled2
    st.markdown("Scaled with digit 2")
    scaled3
    st.markdown("Scaled with digit 3")
    scaled4
    st.markdown("Scaled with digit 4")
    scaled5
    

def scaledPlot():
    st.subheader("Scaling the clusters from 0 to 1 for more accuracy of clusters")
    fig = plt.figure(figsize=(14,7))

    sns.scatterplot(data=scaled1, x=scaled1.Driver_id, y = scaled1['count'], label = 'Cluster 1')
    sns.scatterplot(data=scaled2, x=scaled2.Driver_id, y = scaled2['count'], label = 'Cluster 2')
    sns.scatterplot(data=scaled3, x=scaled3.Driver_id, y = scaled3['count'], label = 'Cluster 3')
    sns.scatterplot(data=scaled4, x=scaled4.Driver_id, y = scaled4['count'], label = 'Cluster 4')
    sns.scatterplot(data=scaled5, x=scaled5.Driver_id, y = scaled5['count'], label = 'Cluster 5')
    #sns.scatterplot(data = c, x = c[:,0], y = c[:,1], label = 'centroid', color = 'black')
    st.pyplot(fig)


# ### Getting the Sum of Squared Errors SSE

# In[11]:

kRange= range(1,10)
sse = []

for k in kRange:
    km = KMeans(n_clusters=k)
    km.fit(scaled[['Driver_id','count']])
    sse.append(km.inertia_)
sse = sse

# In[12]:




# In[13]:

def elbow():
    st.subheader("Plotting sum of squared Elbow plot")
    fig = plt.figure()
    sns.lineplot(x =kRange, y = sse)
    plt.title('SSE ELBOW PLOT',weight = 'bold',color = '#176B87' )
    plt.xlabel('K-Range', color = '#176B87')
    plt.ylabel('SSE', color = '#176B87')
    st.pyplot(fig)
    st.caption("Elbow turning point at 5th K-Range because of five clusters")


# Elbow turning point at 5 because of five clusters

# #### Applying the Gausian model below


# In[14]:


#mdf.head()


# In[15]:


x = pd.DataFrame(mdf['count'])
y = pd.DataFrame(mdf.cluster)


# In[16]:


x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[17]:


model = GaussianNB()
model.fit(x_train, y_train)


# In[18]:


score = model.score(x_test, y_test)


# The score above is very low hence low accuracy

# In[19]:


testing_y = y_test[0:10]


# In[20]:


predictM = model.predict(x_test[0:10])



main()
