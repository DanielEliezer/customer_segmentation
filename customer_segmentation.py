#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ***
# 
# Welcome to my retail customer segmentation project! We are going to analyze the data from the customers of a retail company, and try to understand the behaviour of their customers. Hopefully, we'll find lot of insights to help the company on how to plan their next campaigns, who would be the target of a new product, what are the most important customers, etc.
# 
# The original dataset, that can be found in  <a href="https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis"> this kaggle dataset</a>. It contains information of 2240 customers, with 29 attributes each. These attributes are: 
#  
# **People**
# - ID: Customer's unique identifier
# - Year_Birth: Customer's birth year
# - Education: Customer's education level
# - Marital_Status: Customer's marital status
# - Income: Customer's yearly household income
# - Kidhome: Number of children in customer's household
# - Teenhome: Number of teenagers in customer's household
# - Dt_Customer: Date of customer's enrollment with the company
# - Recency: Number of days since customer's last purchase
# - Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# 
# **Products**
# 
# - MntWines: Amount spent on wine in last 2 years
# - MntFruits: Amount spent on fruits in last 2 years
# - MntMeatProducts: Amount spent on meat in last 2 years
# - MntFishProducts: Amount spent on fish in last 2 years
# - MntSweetProducts: Amount spent on sweets in last 2 years
# - MntGoldProds: Amount spent on gold in last 2 years
# 
# **Promotion**
# 
# - NumDealsPurchases: Number of purchases made with a discount
# - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 
# **Place**
# 
# - NumWebPurchases: Number of purchases made through the company’s website
# - NumCatalogPurchases: Number of purchases made using a catalogue
# - NumStorePurchases: Number of purchases made directly in stores
# - NumWebVisitsMonth: Number of visits to company’s website in the last month

# ## Imports
# ***

# In[1]:


### Imports
import pandas as pd
import numpy as np
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score 
from sklearn.cluster import KMeans

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Reading the data
df = pd.read_csv('marketing_campaign.csv', sep = '\t')

## Renaming columns
df.columns = ['customer_id','year_birth', 'education','marital_status','income', 'kid_home','teen_home','dt_customer','recency', 'mnt_wines', 'mnt_fruits','mnt_meat', 'mnt_fish','mtn_sweet','mnt_gold','num_deals_purchases','num_web_purchases','num_catalog_purchases','num_store_purchases','num_web_visits_month','accepted_cpm3','accepted_cpm4', 'accepted_cpm5', 'accepted_cpm1', 'accepted_cpm2', 'conplain','z_cost_contact', 'z_revenue', 'response']

## Basic info of the dataset
display(df.head(10))
display(df.shape)


# ## Cleaning Data + Feature Engineering 
# ***
# 
# We are going to create new features to the dataset, remove outliers, clean nulls, and change some categorical data.

# In[3]:


## Fill null values with the mean
display(df.isnull().sum())

df = df.fillna(df['income'].mean())

assert (df.isnull().sum().sum() == 0)


# In[4]:


# Feature engineering: Creating new, useful features

# Create a feature with the total spent for the client
df['total_spent'] = df['mnt_wines']  + df['mnt_fruits'] + df['mnt_meat'] + df['mnt_fish'] + df['mtn_sweet'] + df['mnt_gold']

# Create a feature with the % of money spent on essential item (food-related) 
df['pct_essentials'] = (df['mnt_fruits'] + df['mnt_meat'] + df['mnt_fish'] + df['mtn_sweet'])/df['total_spent']

# Create a feature with the number of purchases for each client
df['num_purchases'] = df['num_web_purchases']+df['num_catalog_purchases'] + df['num_store_purchases']+df['num_web_visits_month']

# Create a feature with the number of children
df['num_children'] = df['kid_home'] + df['teen_home']

# Create a feature with the percentage of purchases made with a discount
df['pct_purchases_with_discount'] = df['num_deals_purchases'] / df['num_purchases']
df['pct_purchases_with_discount'].fillna(0, inplace=True)

# Create a feature with the age of the client, and another with the number of days since he's enrolled with the company
df['dt_customer'] = pd.to_datetime(df['dt_customer'])

df['days_since_enrollment'] = ((datetime(2015,1,1) - df['dt_customer']).dt.days)

df['age'] =  2015 - df['year_birth']


# In[5]:


# Group some categories that have a similar meaning 
df['relationship'] = df['marital_status'].replace({'Married':1,'Together':1,
                            'Single':0, 'Divorced':0, 'Widow':0, 'Alone':0, 'Absurd':0,'YOLO':0})

df["education"] = df["education"].replace({'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 
                                           'Graduation':'Graduated', 'Master':'Postgraduate', 'PhD':'Postgraduate'})


# In[6]:


## Removing outliers
df = df[(df['age'] < 90) & (df['income'] < 200000)].reset_index(drop = True)

print("The new dataset have {} rows.".format(df.shape[0]))


# In[7]:


## Since there are just 2 values with 0 purchases and both of them with a low total spent, we'll replace the 0's with 1's.
df.loc[(df['num_purchases'] == 0,'num_purchases')]=1


# In[8]:


## Let's check the relationship between the "offers accepted", and the "percentual of purchases with discount"
df['accepted_sum'] = df['accepted_cpm3']+df['accepted_cpm4']+ df['accepted_cpm5'] + df['accepted_cpm1'] +df['accepted_cpm2']
plt.figure(figsize = (12,8))
plt.title('Percentual of purchases with discount vs number of offers accepted', fontweight = 'bold')
sns.boxplot(data = df, x = 'accepted_sum', y = 'pct_purchases_with_discount')
plt.xlabel('Offers Accepted')
plt.ylabel('Percentual of Purchases with Discount');


# That's really not what we expected: they have an opposite correlation, which doesn't make sense. I'm not going to use these "accepted~" features, because of this incongruence, and because it's not really clear how they work: The description is: "1 if customer accepted the offer in the nst campaign, 0 otherwise". But what happens if the customer didn't receive all the offers (only received 3 of them, for example)? How can we garantee that he saw that offers? What if he received more than 5 offers? Since this features are confusing, I'll not use them, and use the "pct_purchases_with_discount" instead, that will give a similar idea, in a more straightfoward way.

# In[9]:


### Droping columns: Since we are going to use perform clusterization ahead, it's important that we only keep the 
### most important features, since we don't want the clusters to be created based on attributes that 
### are not that important to us



df = df.drop(['num_web_visits_month','recency', 'conplain', 'response', 'customer_id','dt_customer',
              'year_birth','kid_home','teen_home','mnt_wines', 'mnt_fruits','mnt_meat', 'mnt_fish', 
              'mtn_sweet', 'mnt_gold', 'num_web_purchases', 'num_catalog_purchases', 'num_store_purchases',
              'z_cost_contact','z_revenue', 'accepted_cpm3','accepted_cpm4', 'accepted_cpm5', 'accepted_cpm1',
              'accepted_cpm2', 'marital_status', 'accepted_sum'], axis = 1)


# In[10]:


## Purchase behaviour vs Education
plt.figure(figsize = (8,5))
plt.title("Education vs Income", fontweight = 'bold', fontsize = 12)
sns.boxplot(y = df['income'] , x = df['education'], linewidth = 2)
plt.xlabel("")
plt.show()

plt.figure(figsize = (8,5))
plt.title("Education vs Total Spent", fontweight = 'bold', fontsize = 12)
sns.boxplot(y = df['total_spent'] , x = df['education'], linewidth = 2)
plt.xlabel("")
plt.show()

plt.figure(figsize = (8,5))
plt.title("Education vs Number of Purchases", fontweight = 'bold', fontsize = 12)
sns.boxplot(y = df['num_purchases'] , x = df['education'], linewidth = 2)
plt.xlabel("")
plt.show()


# Looking at the purchase behaviour, it's fair to say that graduates and postgraduates have a very similar behaviour. Therefore, we'll put them in the same category.

# In[11]:


df["education"] = df["education"].replace({'Graduated':'Postgraduate'})
df['education'] = df['education'].replace({"Postgraduate": 1})
df['education'] = df['education'].replace({"Undergraduate": 0})
df.rename(columns={'education': 'high_education'}, inplace = True)


# ## Exploratory Data Analysis
# ***
# 
# Let's take a closer look at our remaining features, their distributions, and how they relate to each other.

# In[12]:


# Some descriptive statistics
df.describe()


# In[13]:


## We'll create an auxiliar column, to help us with the visualization. It will be deleted afterwards.
df['age_group'] = df['age']//10*10

cat_columns = ['high_education','num_children', 'relationship', 'age_group']
num_columns = ['income', 'num_deals_purchases','total_spent','num_purchases', 
       'pct_purchases_with_discount', 'days_since_enrollment', 'pct_essentials']


# In[14]:


'''Categorical distributions (even though num_children and age_group are numerical variables,
for practical effects it makes a lot of sense to make their visualization as categorical)
'''

for cat in cat_columns:
    plt.figure(figsize = (8,4))
    plt.title(str.upper(cat), fontweight = 'bold')
    aux = df.groupby(cat).size().reset_index()#.rename({'pct_essentials':'count'}, axis = 1)
    ax = sns.barplot(x=cat, data=aux, y = 0, linewidth=2, edgecolor=".2").set(ylabel="Count", xlabel = None)
    
    plt.show()


# Most of our customers:
# - Are in a relationship
# - Have a high_education (graduated or more)
# - Have 1 children
# - Are between 30~60 years

# In[15]:


## Distribution of numerical variable

for num in num_columns:
    plt.figure(figsize = (8,5))
    plt.title(str.upper(num), fontweight = 'bold')
    ax = sns.histplot(x=num, data=df)
    ax.set_xlabel(" ")
    plt.show()


# In[16]:


## Correlation Matrix
corr= df.corr()

# Getting the Upper Triangle of the co-relation matrix
matrix = np.triu(corr)

fig = plt.figure(figsize=(12,8))
plt.title('Correlation between features', fontsize = 16, fontweight = 'bold')
sns.heatmap(round(df.corr(),3), annot=True, cmap="RdBu", mask = matrix);


# Based on the correlation matrix, we can see a lot of variables that have a strong relationship. Let's take a closer look:

# In[17]:


plt.figure(figsize = (10,6))
plt.title("Total Money Spent vs Number of Children", fontweight = 'bold')
sns.boxplot(data = df, y = 'total_spent', x = 'num_children', linewidth = 2)# interessante
plt.show()

plt.figure(figsize = (10,6))
plt.title("% of purchases with discount vs Number of Children", fontweight = 'bold')
sns.boxplot(data = df, y = 'pct_purchases_with_discount', x = 'num_children', linewidth = 2)
plt.show()

plt.figure(figsize = (10,6))
plt.title("% of Money Spent on essentials vs Numbers of children", fontweight = 'bold')
sns.boxplot(data = df, y = 'pct_essentials', x = 'num_children', linewidth = 2) ### esquisito
plt.show()


plt.figure(figsize = (10,6))
plt.title("% of purchases with discount vs Age Group", fontweight = 'bold')
sns.boxplot(data = df, y = 'pct_purchases_with_discount', x = 'age_group', linewidth = 2)
plt.show()

plt.figure(figsize = (10,6))
plt.title("Income vs Age Group", fontweight = 'bold')
sns.boxplot(data = df, y = 'income', x = 'age_group', linewidth = 2)
plt.show()


plt.figure(figsize = (10,6))
plt.title("Total Spent vs Age Group", fontweight = 'bold')
sns.boxplot(data = df, y = 'total_spent', x = 'age_group', linewidth = 2)
plt.show()

plt.figure(figsize = (10,6))
plt.title("% of Essential items vs Age", fontweight = 'bold')
sns.boxplot(data = df, y = 'pct_essentials', x = 'age_group', linewidth = 2)# interessante
plt.show()

plt.figure(figsize = (10,6))
plt.title("% of Essential items vs Education", fontweight = 'bold')
sns.boxplot(data = df, y = 'pct_essentials', x = 'high_education', linewidth = 2)# interessante
plt.show()

plt.figure(figsize = (10,6))
plt.title("Income vs Number of Purchases", fontweight = 'bold')
sns.scatterplot(data = df, x = 'income', y = 'num_purchases', color = 'green')
plt.show()

plt.figure(figsize = (10,6))
plt.title("Income vs Total Money Spent", fontweight = 'bold')
sns.scatterplot(data = df, x = 'income', y = 'total_spent', color = 'darkred')
plt.show()

plt.figure(figsize = (10,6))
plt.title("Income vs % of purchases with discount", fontweight = 'bold')
sns.scatterplot(data = df, x = 'income', y = 'pct_purchases_with_discount')
plt.show()


df = df.drop(['age_group', 'days_since_enrollment'], axis = 1)


# From the visualizations, we can see some things:
# 
# - People with no children: Spend less money, are less inclined to wait for discounts to buy their items and spend more money on essential items (fruits, fish, meat, sweet).
# 
# - Older people have a higher income
# - People with higher income spend more money, and are less inclined to wait for discounts to buy their items.
# 
# 
# 
# 
# ## Clustering
# ---
# ### Standard Scaler
# 
# Some of the transformations on our data (PCA and Clustering) will envolve measuring distances. That can be problematic, considering that some features have very different ranges. We need to use the StandardScaler, so they can all have a similar range (mostly between -1 to 1).

# In[18]:


### Standardization
sc = StandardScaler()
df_scaled = pd.DataFrame(sc.fit_transform(df), columns = df.columns)
display(df_scaled[:5].T)


# ### Principal Components Analysis (PCA)
# 
# 
# In our dataset, we have a high number of features and a lot of them have a certain level of correlation between each other, and are somewhat redundant. Performing a dimensionality reduction technique is a good option to reduce the number of features, decreasing the computation cost of the algorithm, while maintaining a good amount of information from the original dataset. 
# 
# This step is not mandatory, and is not really clear whether to use PCA before clustering really gives us better results. That is particularly tricky, since clustering is a non-supervisioned algorithm, which makes it harder to assess its performance. 
# 
# Now, performing the PCA, we need to determine how many principal components we'll need to maintain about 70%~80% of variance of the original data. 

# In[19]:


## Determining the number of principal components

variance_ratio = {}
for i in range(1, len(df_scaled.columns)+1):
    pca = PCA(n_components=i)
    pca.fit(df_scaled)
    variance_ratio[f'n_{i}'] = pca.explained_variance_ratio_.sum()*100
    
plt.figure(figsize = (10, 5))
plt.plot(variance_ratio.keys(), variance_ratio.values(), color = 'darkred')
plt.axhline(70, color = 'gray', ls = '--', lw = 1)
plt.title("Variance Ratio (%) vs Number of Principal Components", fontsize = 14, fontweight = 'bold')
plt.ylabel("variance ratio (%)", fontsize = 12)
plt.xlabel("number of principal components", fontsize = 12)
plt.ylim([0, 120])
plt.show()


# With 4 principal components, we have almost 70% of the variance explained. That's a good result.

# In[20]:


## Applying PCA to create a new dataset with 4 Princpal Components.

pca = PCA(n_components = 4, random_state = 123)
pca.fit(df_scaled)
df_pca = pd.DataFrame(pca.transform(df_scaled),
                      columns = (["PC1", "PC2", "PC3", "PC4"]))
df_pca.head(10)


# Now we have a new dataset with the 4 principal components. In the next step, we are going to apply the k-means on this new dataset.

# ## K-Means
# 
# We are going to use the k-means algorithm to create our clusters. First, we'll need to determine the ideal number of clusters. We are going to to that by comparing two techniques: the elbow method and the silhouette score. 
# 
# In the elbow method, we plot the WCSS vs the number of clusters. The WCSS is the sum of squared distance between each point and the centroid in a cluster. Naturally, as we increase the number of clusters, this value will always get smaller. 
# 
# What we are looking for is a value of N that, from that point forward, the decrease of WCSS won't be very significative.

# In[21]:


## Determining the ideal number of clusters: Elbow method
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(df_pca)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker='o', color = 'tab:orange')
plt.title('Elbow Method', fontsize = 14, fontweight = 'bold')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


# Taking a look at the curve, it's not really clear what is the ideal number of clusters. Let's double check it with the silhouette scores. From wikipedia: *The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation).*

# In[22]:


silhouette_scores = []

for i in range(2,10):
    m1 = KMeans(n_clusters=i, random_state=123)
    c = m1.fit_predict(df_pca)
    silhouette_scores.append(silhouette_score(df_pca, m1.fit_predict(df_pca)))

plt.figure(figsize = (8,5))
plt.bar(range(2,10), silhouette_scores, color = 'darkgreen')
plt.title("Silhouette Scores vs Number of Clusters", fontsize = 14, fontweight = 'bold')
plt.xlabel('Number of clusters', fontsize = 14) 
plt.ylabel('Silhouette Scores', fontsize = 14)
plt.ylim(0,0.35)
for a, b in enumerate(silhouette_scores):
    plt.text(a + 1.7, b + 0.005, str(round(b,3)), color='black', fontweight='bold')
plt.show()


# That settles: We are going to use the K-Means with 3 clusters, and add the results to the original dataset.

# In[23]:


kmeans = KMeans(n_clusters=3, random_state=123).fit(df_pca)
pred = kmeans.predict(df_pca)
df['cluster'] = pred + 1
df.head()


# ## Analyzing the clusters
# 
# Now, let's compare the clusters:

# In[24]:


aux = df.cluster.value_counts()/len(df.count())
plt.title('Number of Customers per Cluster', fontweight = 'bold')
aux = df.groupby('cluster').size().reset_index()#.rename({'pct_essentials':'count'}, axis = 1)
ax = sns.barplot(x='cluster', data=aux, y = 0, linewidth=2, edgecolor=".2",  palette = 'Set1').set(ylabel="Count", xlabel = None)


# In[25]:


# Creating an auxiliar dataset to help the evaluation
clusters_evaluation = df.groupby('cluster').mean().T
clusters_evaluation['mean_dataset'] = df.mean().T
clusters_evaluation


# In[26]:


plt.figure(figsize = (8,8))
clusters_evaluation.drop('mean_dataset', axis = 1).style.background_gradient(cmap='RdYlGn',axis=1)


# From the table above, we can have a good idea of the differences between the clusters. At first glance, we can see that the income is very important to divide the users: In the cluster 1, we have a lot of customers with medium income, the cluster 2 we have people with lower-income, and the 3, with a higher income. 
# 
# Let's make some more visualizations, to get some more insights.

# In[27]:


## Categorical Visualizations

cat_columns = ['high_education','num_children', 'relationship']
num_columns = ['income','total_spent','num_purchases', 
       'pct_purchases_with_discount', 'age', 'pct_essentials']

for cat in cat_columns:
    plt.figure(figsize = (8,5))
    plt.title(str.upper(cat), fontweight = 'bold')
    df_aux = df.groupby('cluster')[cat].value_counts(normalize=True).mul(100).rename('percent').reset_index()
    sns.barplot(data = df_aux, x='cluster',y='percent',hue=cat, palette = 'Paired',linewidth=2, edgecolor=".2");
    plt.show()    


# **Notes:**
# 
# - Most of the users without high-education falls into the cluster 2
# - Pretty much everyone of the cluster 1 have at least 1 children.
# - Pretty much everyone of the cluster 3 have 0 or 1 children

# In[28]:


## Numerical Variables

for num in num_columns:
    plt.figure(figsize = (8,5))
    plt.title(str.upper(num), fontweight = 'bold')
    ax = sns.boxplot(y=num, x = 'cluster', data=df, palette = 'Set1', linewidth = 2)#.set(ylabel="Count", xlabel = None)
    plt.show()


# #### Notes:
# 
# ##### <ins>Income:<ins>
# - 1: High Income 
# - 2: Low Income
# - 3: Medium Income
# 
# #### <ins>Total Spent:<ins> 
# - 1: high spent
# - 2: low spent
# - 3: medium spent
# 
# #### <ins>Percentage of Items bought with discount:<ins>
# - 3: Buys a lot of items with discount
# - 1: Buys a lot of items without discount
# 
# #### <ins>Percentage of essential items bought:<ins>
# - 3: Buys a lot of non-essential items (wine + gold)
# 
# #### <ins>Age:<ins>
# - 2: Are considerably younger than the rest 

# 
#     
# Other important visualizations:

# In[29]:


plt.figure(figsize = (9,5))
plt.title('Percentage of items with discount vs Income', fontweight = 'bold')
sns.scatterplot(data = df, x = 'income', y = 'pct_purchases_with_discount', hue = 'cluster', palette = 'Set1')
plt.show()

plt.figure(figsize = (9,5))
plt.title('Percentage of essential items bought vs Income', fontweight = 'bold')
sns.scatterplot(data = df, x = 'income', y = 'pct_essentials', hue = 'cluster', palette = 'Set1')
plt.show()

plt.figure(figsize = (9,5))
plt.title('Number of purchases vs Income', fontweight = 'bold')
sns.scatterplot(data = df, x = 'income', y = 'num_purchases', hue = 'cluster', palette = 'Set1')
plt.show()

plt.figure(figsize = (9,5))
plt.title('Total Money Spent vs Income', fontweight = 'bold')
sns.scatterplot(data = df, x = 'income', y = 'total_spent', hue = 'cluster', palette = 'Set1')
plt.show()


# ## Conclusions:
# ***
# 
# 
# ### Cluster 1:
# 
# 
# - About 36% of the customers
# - High income group
# - Almost everybody have 0 or 1 children.
# - Spend a lot of money
# - Don't buy a lot of things with discount
# 
# 
# ### Cluster 2:
# 
# - Bigger cluster (about 44%)
# - Smaller income group
# - Don't spend a lot of money
# - Don't make a lot opf
# - Contains most of the people without a high-education
# - Is a little younger than the average of the dataset
# 
# ### Cluster 3:
# 
# - Smaller part of the customers (about 20%)
# - Medium income group
# - Everyone have at least 1 children
# - Are very sensitive to items with discount
# - Usually buys non-essential items (wine + gold)
# 
# 
# From this clusters, we are a step closer to understand our customers behaviour. The more actionable information is that relation between the customers and the % of items bought with discount:
# 
# - We could see that people in the cluster 3 buys a lot of non-essential items, and are very sensitive to discount. 
# - On the other hand, people in the cluster 1 doesn't respond very well to discounts.
# 
# Therefore, it **might be a good idea to concentrate efforts to send special offers to the people in the cluster 3 (and don't send to cluster 1).**
