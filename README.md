**Data Cleaning**

In this work we are trying to clean our datasets using wide range of techniques. We start off by importing our dataset, identify the missing values, learn about scaling and normalizing techniques, and end with parsing and encoding tasks. 

For this project we are using the airbnb dataset on listing of houses in NYC. 


```python
pip install plotly_express
```


```python
# libraries we'll use
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import plotly_express as px
import os
import gc
```


```python
# read your data
df = pd.read_csv(r"C:\Users\afard\Downloads\Airbnb_Open_Data.csv")
```


```python
df.head()
```


```python
# sometimes it is better to look at some random observations rahter than just the top 5
df.sample(10)
```


```python
df.shape
```


```python
df.info()
```


```python
# we can simply see the total number of missing observations in each features

missing_df_count = df.isnull().sum()
missing_df_count
```


```python
# I like to see them in a sorted way 

df.isnull().sum().sort_values(ascending=False)
```


```python
# to get an overview of what 5 of our dataset is missing, we can use the following

total_obs = np.product(df.shape)
total_missing = missing_df_count.sum()

# percent of data that is missing
(total_missing/total_obs) * 100
```

**7% of our entire observations are missing** 

Depending on your work, this could be a lot of not much. I think I can live with it

out of 102599 observations, it seems like the variable *licence* is missing in 102597 cases. So we will definitely drop this variable, however, I like to look at the cases that actually have this variables (2 obs)


```python
df.loc[~df.license.isnull()]
```

**Christina** is the only licenced airbnb host in the entire NYC. Nice!

Another interesting thing is that the same place (given the NAME and address) is registered using two id numbers. This is a red flag (even though we were not looking for one), that there are potential duplications in our dataset. 


```python
# Let's get rid of licence column

df=df.drop(columns=['license'])

# you can use this as well
# df.drop('license', axis=1, inplace=True)
```

**Missing values: to drop or not?**

I think before we can answer to this important philosophical question, we need to know why the data is missing? (out of all the questions in the current state of world)

Essentially, the question is whether the data **does not exist** (like house rules for the host who does not have any rules), or
is it because it **does not recorded** (Like country variable: I guess we all can agree that NYC is in Japan). 

Knowing the answer to this question can help us to either drop the variable (Impute the bastard!) or try to replace it (i just spent 2 hours to read the literature on the best way to replace a missing value and do not have any good answer). But if you feel like reading 100 pages of a paper check this paper out

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4106794

Make sure to stay some time and study your dataset before choosing either of these options. Some datasets have documentation that can help you understand the nature of the data, hence, why is it missing?



```python
# Let's drop any duplicate observations based on id and Name- Before that let's identify those obs 
# I want to chose duplication based on id, lat, and long (which represents the address of the property).

print(df[['id', 'lat', 'long']].dtypes) #it is easier if our variables are not object or string
```


```python
print(df.duplicated(subset=['id', 'lat', 'long'], keep=False))
```


```python
# identify duplicate rows based on ID and Name in a subset of columns
dup_mask = df.duplicated(subset=['id', 'host id', 'lat', 'long'], keep=False)

# filter the original dataframe to show only the duplicate rows
dup_df = df[dup_mask].sort_values(by=['id', 'lat', 'long'])

# print the result
print(dup_df)
```


```python
# there are 1082 observations with duplication values. let's delete the duplications

# Sort the DataFrame by the columns used to identify duplicates
df = df.sort_values(by=['id', 'host id', 'lat', 'long'])

# Drop duplicates, keeping the first occurrence of each
df = df.drop_duplicates(subset=['id', 'host id', 'lat', 'long'], keep='first')
```


```python
df.shape
```


```python
df.isnull().sum().sort_values(ascending=False)
```


```python
df.head(3)
```


```python
# Next, let's replace the missing 'house_rules' column with the word 'blank'

df.loc[df.house_rules.isnull(), 'house_rules'] = 'blank'
```


```python
# Our next varable with many missing is 'last review'

df['last review']
```


```python
# since it is a date, let's start by fixing the date format

df['last review'] = pd.to_datetime(df['last review'])
```


```python
# let's check the min and max for this variable so that we can change the missing accordingly

df['last review'].min(), df['last review'].max()
```


```python
# BINGO! we find another flaw in our dataset. the year obviously cannot be 2058. let's find other suckers that have a wrong date

df[df['last review'].apply(lambda x: x.year) > 2022]
```


```python
# we can change these reviews with a median or mean of the date if we want to keep them. I rather just drop them. if you wanted
# to keep them use below code

#df.loc[df[df['last review'].apply(lambda x: x.year) > 2022].index, 'last review'] = df['last review'].median()

df = df.drop(df[df['last review'] > pd.Timestamp('2022-12-31')].index)
```


```python
df['last review'].min(), df['last review'].max()
```


```python
# Now let's impute the null values to the minimum date in the dataset

df.loc[df['last review'].isnull(), 'last review'] = df['last review'].median()
```


```python
# Let's check the null counts once again

df.isnull().sum().sort_values(ascending=False)
```


```python
# Next, let's explore the 'reviews per month' column

fig = px.histogram(df, x='reviews per month', log_y=True, )
fig.show()
```

**How to deal with right skewed data like this?**

*According to the ChatGPT*

If a dataset is severely right-skewed, meaning that the majority of the data is clustered towards the lower end of the range and there are a few high values that are far from the rest, then replacing missing values with the median may not be the best option.

In such cases, it's important to analyze the data and understand the reasons behind the skewness. If the skewness is due to outliers or extreme values, then replacing missing values with the median can be a good option. However, if the skewness is due to the nature of the data, such as income or age, then replacing missing values with the median may not be the best approach.

Other options for dealing with missing values in a severely skewed dataset include:

Using mean or mode: In some cases, the mean or mode may be a better representation of the data than the median.
Imputing values using a regression model: This method involves using a regression model to predict the missing values based on the values of other variables.
Deleting the missing values: If the proportion of missing values is relatively small, then it may be appropriate to simply delete the missing values. However, this method can lead to a reduction in sample size and potentially bias the results.

**the histogram shows the existance of outliers, so we can replace the missing with median**


```python
median_A = df['reviews per month'].median()
print("Median of column 'A':", median_A)
```


```python
df.loc[df['reviews per month'].isnull(), 'reviews per month'] = 0.74
```


```python
# This is one of my favorite piece of code. How many unique values exists in each columns?

unique_counts = df.nunique()
print(unique_counts)
```


```python
# for a variable with limited number of unique obs, we can see the frequency of each


df.host_identity_verified.value_counts()
```


```python
df.cancellation_policy.value_counts()
```


```python
df['neighbourhood group'].value_counts()
```

**did you see what just happened?** the different variable names require different ways of getting value counts


```python
# In the next couple of code blocks we try to make some changes based on our previous observations 

# Impute NAME column with 'blank'
df.loc[df['NAME'].isnull(), 'NAME'] = 'blank'   

# Impute host id with 0
df.loc[df['host id'].isnull(), 'host id'] = 0  

# Impute host_identity_verified with 'unconfirmed'
df.loc[df['host_identity_verified'].isnull(), 'host_identity_verified'] = 'unconfirmed'

# Impute host name with 'blank'
df.loc[df['host name'].isnull(), 'host name'] = 'blank'
```


```python
# Fix the spellings of manhattan and brooklyn in column 'neighbourhood group' and impute missing using lat/long
df.loc[df['neighbourhood group']=='manhatan', 'neighbourhood group'] = 'Manhattan'
df.loc[df['neighbourhood group']=='brookln', 'neighbourhood group'] = 'Brooklyn'
```


```python
# Let's drop 'country' and 'country code' because they have zero variability
df.drop(['country', 'country code'], axis=1, inplace=True)
```


```python
# since our lat and long variables have minimal missing obs, we use them to fill out our neighborhood variable
!pip install geopy
```


```python
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="MyApp")
```


```python
# Check the missing neighbourhood rows

df.loc[df.neighbourhood.isnull()]
```


```python
# Let's define a function to accept coordinates and return suburb name
def loc_from_coord(lat, long):
    location = geolocator.reverse(str(lat)+","+str(long))
    return location.raw['address'].get('road', '')

# Let's test the function
temp = df.loc[df.neighbourhood.isnull()].copy()
print(loc_from_coord(temp.iloc[0].lat, temp.iloc[0].long))
```


```python
# So the sample worked, now we impute all the missing neighbourhood data

idx = df.loc[df.neighbourhood.isnull()].index
df.loc[idx, 'neighbourhood'] = df.loc[idx].apply(lambda x: \
                                                loc_from_coord(x.lat, x.long), axis=1)
```


```python
del temp
```


```python
# Let's check whether the imputation worked or not
df.loc[idx].head()
```


```python
# Okay, so that worked. Now we'll impute 'neighbourhood group'
# Let's check the rows
df.loc[df['neighbourhood group'].isnull()]
```


```python
# It's a long list. Let's make a function to convert the coordinates to neighbourhood group

def neigh_from_coord(lat,long):
    location = geolocator.reverse(str(lat)+","+str(long))
    return location.raw['address'].get('suburb', '')
```


```python
# Let's check a sample
idx = df.loc[df['neighbourhood group'].isnull()].index
print(neigh_from_coord(df.loc[idx].iloc[0].lat, df.loc[idx].iloc[0].long))
```


```python
# So the sample worked, now we impute all the missing neighbourhood group data

df.loc[idx, 'neighbourhood group'] = df.loc[idx].apply(lambda x: neigh_from_coord(x.lat, x.long), 
                                                       axis=1)
```


```python
# Let's check whether the imputation worked or not - choose a subset of dataframe
df.loc[idx]
```


```python
# Okay, so that worked. Let's now try to impute lat/long using neighbourhood group and neighbourhood
df.loc[df.lat.isnull()]
```


```python
# Let's collect the indices as earlier (lat and long are missing together)
idx = df.loc[df.lat.isnull()].index

# Now we define a function to accept the location and return latitude and longitude
def lat_from_loc(loc):
    location = geolocator.geocode(loc)
    return location.latitude

def long_from_loc(loc):
    location = geolocator.geocode(loc)
    return location.longitude

# Let's test a sample case
print(lat_from_loc(df.loc[idx].iloc[0].neighbourhood), long_from_loc(df.loc[idx].iloc[0].neighbourhood))
```


```python
# Now that looks pretty good, let's impute all the missing coordinates (used both neighbourhood and 
# neighbourhood group because there can be multiple suburbs with same name, such as, Elmhurst is also in IL)

df.loc[idx, 'lat'] = df.loc[idx].apply(lambda x: lat_from_loc(x.neighbourhood+', '+x['neighbourhood group']), axis=1)
df.loc[idx, 'long'] = df.loc[idx].apply(lambda x: long_from_loc(x.neighbourhood+', '+x['neighbourhood group']), axis=1)
```


```python
df.loc[idx]
```


```python
# So now we'll check for null values again, data cleaning is always a long process, and I'm not the most efficient

df.isnull().sum().sort_values(ascending=False)
```


```python
# Check 'availability 365'
fig = px.histogram(df, x='availability 365')
fig.show()
```


```python
fig = px.box(df, y='availability 365')
fig.show()
```


```python
median_A = df['availability 365'].median()
print("availability 365 :", median_A)
```


```python
# We impute availability 365 with median value 96
df['availability 365'] = df['availability 365'].fillna(96)
```


```python
# Check minimum nights
fig = px.histogram(df, x='minimum nights')
fig.show()
```


```python
# there are many things wrong with this variable. negative numbers for minimum night? and some really high numbers

df['minimum nights'].min(), df['minimum nights'].max()
```


```python
# We'll take log normal
fig = px.histogram(df, x='minimum nights', log_y=True)
fig.show()
```


```python
# Let's clip the data between 0 and 13, the upper fence (Q3 + 1.5 * IQR)
df['minimum nights'].clip(lower=0, upper=13, inplace=True)
fig = px.histogram(df, x='minimum nights', log_y=True)
fig.show()
```


```python
fig = px.box(df, y='minimum nights', log_y=True)
fig.show()
```


```python
# Let's impute the 'minimum nights' feature with the median 3

df['minimum nights'] = df['minimum nights'].fillna(3)
```


```python
# Check the price feature
# First we'll convert price from object to numeric
import re

idx = df.loc[~df.price.isnull()].index
df.loc[idx, 'price'] = df.loc[idx].apply(lambda x: re.sub(r'\D', '', x.price), axis=1)
df.loc[idx, 'price'] = pd.to_numeric(df['price'])
```


```python
type(df.price[0])
```


```python
fig = px.histogram(df, x='price')
fig.show()
```


```python
# since the mean and median are close, we use mean to replace missing variables

df.price.fillna(df.price.mean(), inplace=True)
```


```python
df.isnull().sum().sort_values(ascending=False)
```


```python
# Check service fee
df['service fee'].dtype
```


```python
# We'll give same treatment to service fee as price
idx = df.loc[~df['service fee'].isnull()].index
df.loc[idx, 'service fee'] = df.loc[idx].apply(lambda x: re.sub(r'\D', '', x['service fee']), axis=1)
df.loc[idx, 'service fee'] = pd.to_numeric(df['service fee'])
```


```python
type(df['service fee'][0])
```


```python
fig = px.histogram(df, x='service fee')
fig.show()
```


```python
df['service fee'].mean()
```


```python
# Let's impute the service fee with mean
df['service fee'].fillna(df['service fee'].mean(), inplace=True)
```


```python
# Let us take the following cumulative actions:
# 1. Check the distribution and impute review rate number
# 2. Check the distribution and impute Construction year
# 3. Check the distribution and impute number of reviews
# 4. Check the distribution and impute calculated host listings count
# 5. Check the unique values and impute instant_bookable
# 6. Check the unique values and impute cancellation_policy
```


```python
# review rate number
fig = px.box(df, y='review rate number')
fig.show()
```


```python
# It's a modest distribution, we'll impute with median
df['review rate number'].fillna(df['review rate number'].median(), inplace=True)
```


```python
# Construction year
fig = px.box(df, y='Construction year')
fig.show()
```


```python
# Impute with median
df['Construction year'].fillna(df['Construction year'].median(), inplace=True)
```


```python
# number of reviews
fig = px.histogram(df, x='number of reviews')
fig.show()
```


```python
# It has a heavy right skew, let's check log transform
fig = px.histogram(df, x='number of reviews', log_y=True)
fig.show()
```


```python
fig = px.box(df, y='number of reviews', log_y=True)
fig.show()
```


```python
# Impute with median
df['number of reviews'].fillna(df['number of reviews'].median(), inplace=True)
```


```python
# calculated host listings count
fig = px.histogram(df, x='calculated host listings count')
fig.show()
```


```python
# It has a heavy right skew, let's check log transform
fig = px.histogram(df, x='calculated host listings count', log_y=True)
fig.show()
```


```python
# Impute with median
df['calculated host listings count'].fillna(df['calculated host listings count'].median(), inplace=True)
```


```python
# Check the unique values and impute instant_bookable
df.instant_bookable.value_counts()
```


```python
# Giving the host benefit of doubt, impute the column with True
df.instant_bookable.fillna(True, inplace=True)
```


```python
# Check the unique values and impute cancellation_policy
df.cancellation_policy.value_counts(0)
```


```python
# Again,giving host benefit of doubt, impute with 'moderate'
df.cancellation_policy.fillna('moderate', inplace=True)
```


```python
# Final check for null values
df.isnull().sum()
```


```python
df['neighbourhood group'].value_counts()
```


```python
# Convert the single occurance of 'The Bronx'
df.loc[df['neighbourhood group']=='The Bronx', 'neighbourhood group']='Bronx'
```


```python
# Convert the column headers for later ease of usage
df.columns = df.columns.str.lower().str.replace(' ','_')
df.head(1)
```


```python
# let's export our clean data 
df.to_csv('airbnb_nyc_clean.csv', index=False)
```


```python
os.listdir()
```
