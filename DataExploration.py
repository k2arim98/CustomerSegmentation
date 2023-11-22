#Required Packages and Data
import math
import random
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import timedelta
#%matplotlib inline

sns.set_style("darkgrid")

from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors

from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)

df = pd.read_excel("OnlineRetail.xlsx")
df.head()

#Handling Missing Values
df.isnull().sum()
df.drop(df[df["CustomerID"].isnull()].index, axis=0, inplace=True)
df["Description"] = df["Description"].fillna("")
df.isnull().sum().sum()

#Treating Canceled Invoices
size_before = len(df)
neg_quantity = df[df["Quantity"] < 0][["CustomerID", "StockCode", "Quantity"]].sort_values("Quantity")
print(f"Negative Quantity: {len(neg_quantity)}")

filtered = df[df["CustomerID"].isin(neg_quantity["CustomerID"])]
filtered = filtered[filtered["StockCode"].isin(neg_quantity["StockCode"])]

pos_counters = []
for idx, series in neg_quantity.iterrows():
    customer = series["CustomerID"]
    code = series["StockCode"]
    quantity = -1 * series["Quantity"]
    counterpart = filtered[(filtered["CustomerID"] == customer) & (filtered["StockCode"] == code) & (filtered["Quantity"] == quantity)]
    pos_counters.extend(counterpart.index.to_list())

to_drop = neg_quantity.index.to_list() + pos_counters
df.drop(to_drop, axis=0, inplace=True)
print(f"Removed {size_before - len(df)} rows from the dataset")

#value of UnitPrice is 0
df.drop(df[df["UnitPrice"] == 0].index, axis=0, inplace=True)

#extracting time related features from InvoiceDate

df["InvoiceDateDay"] = df["InvoiceDate"].dt.date
df["InvoiceDateTime"] = df["InvoiceDate"].dt.time
df["InvoiceYear"] = df["InvoiceDate"].dt.year
df["InvoiceMonth"] = df["InvoiceDate"].dt.month
df["InvoiceMonthName"] = df["InvoiceDate"].dt.month_name()
df["InvoiceDay"] = df["InvoiceDate"].dt.day
df["InvoiceDayName"] = df["InvoiceDate"].dt.day_name()
df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.day_of_week
df["InvoiceHour"] = df["InvoiceDate"].dt.hour

df["TotalValue"] = df["Quantity"] * df["UnitPrice"]

#RFM Analysis(group the customers into meaningfull groups)
ref_date = df["InvoiceDateDay"].max() + timedelta(days=1)

df_customers = df.groupby("CustomerID").agg({
    "InvoiceDateDay": lambda x : (ref_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalValue": "sum"
}).rename(columns={
    "InvoiceDateDay": "Recency",
    "InvoiceNo": "Frequency",
    "TotalValue": "MonetaryValue"
})

df_customers.head(10)

#Removing Outliers
n_cols = len(df_customers.columns)
fig, axes = plt.subplots(n_cols, 2, figsize=(16, n_cols * 4))

for i, col in enumerate(df_customers.columns):
    sns.boxplot(data=df_customers, y=col, ax=axes[i][0])
    sns.histplot(data=df_customers, x=col, kde=True, ax=axes[i][1])

fig.show()

def remove_outliers(df, col, threshold=1.5):
    Q1 = np.quantile(df[col], .25)
    Q3 = np.quantile(df[col], .75)
    IQR = Q3 - Q1
    df.drop(df[(df[col] < (Q1 - threshold * IQR)) | (df[col] > (Q3 + threshold * IQR))].index, axis=0, inplace=True)
    
    return df

for col in df_customers.columns:
    size_before = len(df_customers)
    df_customers = remove_outliers(df_customers, col)
    print(f"Removed {size_before - len(df_customers)} outliers from {col}")
    
n_cols = len(df_customers.columns)
fig, axes = plt.subplots(n_cols, 2, figsize=(16, n_cols * 4))

for i, col in enumerate(df_customers.columns):
    sns.boxplot(data=df_customers, y=col, ax=axes[i][0])
    sns.histplot(data=df_customers, x=col, kde=True, ax=axes[i][1])

fig.show()

#Exploring the DataSet
##Revenue by Country

country_revenue = df.groupby("Country")["TotalValue"].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(12, 4))
plt.title("Total Revenue by Country")
sns.barplot(data=country_revenue, x="Country", y="TotalValue")
plt.ylabel("Total Revenue")
plt.xticks(rotation=90)
plt.show()

##Revenue by Month of the Year
revenue_month = df.groupby(["InvoiceMonth", "InvoiceMonthName"])["TotalValue"].sum().reset_index()
plt.figure(figsize=(12, 4))
plt.title("Total Revenue by Month of the Year")
sns.barplot(data=revenue_month, x="InvoiceMonthName", y="TotalValue")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.show()

##Revenue by Day of the Week
revenue_day = df.groupby(["InvoiceWeekOfYear", "InvoiceDayOfWeek", "InvoiceDayName"])["TotalValue"].sum().reset_index()
revenue_day.groupby(["InvoiceDayOfWeek", "InvoiceDayName"])["TotalValue"].mean().reset_index()
plt.title("Average Revenue by Month of the Year")
sns.barplot(data=revenue_day, x="InvoiceDayName", y="TotalValue")
plt.xlabel("Day of Week")
plt.ylabel("Total Revenue")
plt.show()

##Top 10 Customers by Revenue
top_10_customers = df_customers["MonetaryValue"].sort_values(ascending=False).head(10)
top_10_customers

##Customers RFM
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Features Distribution")
sns.histplot(df_customers["Recency"], bins=50, ax=ax[0], kde=True)
sns.histplot(df_customers["Frequency"], bins=50, ax=ax[1], kde=True)
sns.histplot(df_customers["MonetaryValue"], bins=50, ax=ax[2], kde=True)
fig.show()