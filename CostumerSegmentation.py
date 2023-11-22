import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from tqdm import trange

#Importing Data Set
df = pd.read_excel('OnlineRetail.xlsx')
df.head()

#Data Pre-Processing
df['CustomerID'] = df['CustomerID'].fillna(-1).astype(int).astype('str').replace('-1', 'Unknown')
df['Description'] = df['Description'].str.strip()
df['On Credit'] = df['InvoiceNo'].str.contains('C')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['ItemTotal'] = df['Quantity'] * df['UnitPrice']

# Discarding Unknown Customers or product returns for simplicity of concept
df = df[(df['CustomerID'] != 'Unknown') & (df['UnitPrice'] > 0)].reset_index(drop=True).copy()
df.head()

#Extracting Product Feed
feed_df = df.groupby(['StockCode', 'Country', ]).agg({'Description': 'first', 'UnitPrice': 'mean'}).reset_index().copy()
feed_df.head()

#Extracting Invoices
invoices_df = df.groupby(['InvoiceNo']).agg(
    {
        'StockCode': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': ['mean', 'max', 'min'],
        'ItemTotal': 'sum'
    }
)
invoices_df.columns = ['_'.join(col).strip()
                       for col in invoices_df.columns.values]

#Discarding Outliers
[lv, uv] = invoices_df['Quantity_sum'].quantile([0.05, 0.95]).values

invoices_df = invoices_df[(invoices_df['Quantity_sum'] >= lv) & (
    invoices_df['Quantity_sum'] <= uv)].reset_index().copy()
invoices_df.head()

#Isolating Creating Customer Feature Set
customer_df = df.sort_values('InvoiceDate').groupby('CustomerID').agg(
    {
        'InvoiceDate': ['first', 'last'],
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': 'mean',
    }
)

customer_df.columns = ['FirstPurchaseDate', 'LastPurchaseDate',
                       'TotalOrders', 'TotalQuantity', 'AveragePurchasePrice']

customer_df['DaysToDate'] = (
    customer_df['LastPurchaseDate'] - customer_df['FirstPurchaseDate']).dt.days + 1


# Ignoring one off customers
customer_df = customer_df[customer_df['DaysToDate'] > 1].copy()

customer_df.head()

#Merging Customers with Invoice data and Adding some features
invoices_df = invoices_df.merge(
    df[['CustomerID', 'InvoiceNo']],
    on='InvoiceNo'
)

invoices_df = invoices_df.groupby('CustomerID').agg(
    {
        'InvoiceNo': 'nunique',
        'StockCode_nunique': 'sum',
        'Quantity_sum': 'sum',
        'UnitPrice_mean': 'mean',
        'UnitPrice_max': 'mean',
        'UnitPrice_min': 'mean',
        'ItemTotal_sum': 'mean',
    }
)


customer_df = customer_df.merge(invoices_df, left_index=True, right_index=True)
customer_df['PurchaseFrequency'] = customer_df['TotalOrders'] / \
    customer_df['DaysToDate']

customer_df['AverageInvoiceItems'] = customer_df['TotalQuantity'] / \
    customer_df['TotalOrders']

customer_df = customer_df[customer_df['TotalQuantity'] > 0].copy()

customer_df.drop(columns=['FirstPurchaseDate',
                 'LastPurchaseDate'], inplace=True)
customer_df.head()

#Scaling Data, Clustering, Finding Optimal Clusters
scaler = StandardScaler()

cdf = scaler.fit_transform(customer_df)
wcss = []

for each in trange(1, 15):
    kmeans = KMeans(n_clusters=each, n_init='auto', max_iter=1000, random_state=42)
    kmeans.fit(cdf)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 15), wcss)
plt.title('Elbow plot')
plt.xlabel("Number of k value (each)")
plt.ylabel("WCSS")
plt.show()

#Clustering with Optimal Number of Clusters
n_clusers = 6
model = KMeans(
    n_clusters=n_clusers,
    n_init='auto',
    max_iter=1000,
    random_state=42
)
cluster_k = model.fit_predict(cdf)


with_km_df = pd.DataFrame(
    cdf, columns=customer_df.columns, index=customer_df.index)
with_km_df['cluster'] = cluster_k
with_km_df['cluster'].value_counts(ascending=True)

#Mapping Products to Clusters
customer_df = customer_df.merge(
    with_km_df['cluster'], left_index=True, right_index=True).reset_index()

clustered_products_df = df.merge(customer_df[['CustomerID', 'cluster']], on='CustomerID')[
    ['StockCode', 'cluster']].drop_duplicates()
feed_df = feed_df.merge(clustered_products_df)

print(feed_df[['StockCode', 'cluster']].drop_duplicates()['cluster'].value_counts())
feed_df.head()

#Visualizing Product attributes
rows= []
for i in range(n_clusers):
    rows.append([i] + feed_df[feed_df['cluster'] == i].UnitPrice.quantile(
        [0.05, .25, .5, .75, 0.95]).values.tolist())

pdf = pd.DataFrame(rows, columns=['cluster', '5%', '25%', '50%', '75%', '95%'])

pdf

#Visualizing Customer to Product Mappings
k_customers_df = with_km_df['cluster'].value_counts().to_frame('Customers')
k_feed_df = feed_df[['StockCode', 'cluster']].drop_duplicates().groupby('cluster').size().to_frame('Products')
c2p_df = k_customers_df.merge(k_feed_df, left_index=True, right_index=True)

c2p_df.sort_index()