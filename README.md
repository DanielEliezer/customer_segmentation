# customer_segmentation

**- Description and Motivation**
This project analyzes the data from the customers of a retail company, and try to understand the behaviour of their customers. That can give the company a lot of insights on how to plan their next campaigns, who would be the target of a new product, what are the most important customers, etc.

Link to the original dataset: https://www.kaggle.com/imakash3011/customer-personality-analysis

**- Main Steps**

• Introduction

• Preprocessing: Feature Engineering, removing nulls, removing outliers, changing categorical features and removing irrelevants variables.

• Exploratory Data Analysis (EDA): Some plots to understand our features distribution, and how they relate to each other

• Clustering: Scaling the features, Principal components analysis (PCA) and K-Means

• Evaluation of the results: Who are the customers assigned to each customer? 

**- Conclusions:**

**Cluster 1:**
- Smaller part of the customers (about 20%)
- Medium income group
- Everyone have at least 1 children
- Are very sensitive to items with discount
- Usually buys non-essential items (wine + gold)

**Cluster 2:**
- Bigger cluster (about 44%)
- Smaller income group
- Don't spend a lot of money
- Contains most of the people without a high-education
- Is a little younger than the average of the dataset

**Cluster 3:**
- About 36% of the customers
- High income group
- Almost everybody have 0 or 1 children.
- Spend a lot of money
- Don't buy a lot of things with discount

The more actionable information is that relation between the customers and the % of items bought with discount:

We could see that people in the cluster 1 buys a lot of non-essential items, and are very sensitive to discount.
On the other hand, people in the cluster 3 doesn't respond very well to discounts.
Therefore, **it might be a good idea to concentrate efforts to send special offers to the people in the cluster 1 (and don't send to cluster 3).**

Also, customers from the **cluster 3** spend more money, therefore **are our most valuable customers**. It's worth doing an extra efort to make them happy!

**- Files in the repository:**

• marketing_campaign.csv: A dataset with 2040 rows, and each one of them stores information about a specific customer.
• customer_segmentation.ipynb: The notebook of the project
• customer_segmentation.py. The project as a python file.

**- Libraries used:** Pandas, Numpy, Datetime, Seaborn, Matplotlib, Sklearn
