from AJ_draw import disegna as ds
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from classification_lib import marketing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import streamlit as st

# tx_data = pd.read_csv('data/OnlineRetail.csv', encoding="ISO-8859-1")
# tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
# tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']
# tx_data = tx_data[tx_data['Country'] == 'United Kingdom']
# tx_data = tx_data.drop(['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'Country'], axis = 1)
# tx_data = tx_data.dropna()
# tx_data.columns = ['Date', 'ID', 'Revenue']
# st.write(tx_data.head(10))
# tx_data.to_csv('marketing1.csv', index = False)

# tx_data = pd.read_csv('marketing1.csv')#, encoding="ISO-8859-1")
# tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
# st.write(tx_data)

#function for ordering cluster numbers
#visto che kmean quando crea le classificazioni non e' ordinato secondo nessun criterio particolare
#con questa funzione il database rinomina le classi in modo che siano ordinate secondo la recency
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


def rec_freq_rev(tx_data):
    #create a generic user dataframe to keep CustomerID and new segmentation scores

    tx_user = pd.DataFrame(tx_data['ID'].unique())
    tx_user.columns = ['ID']

    # st.title('Recency')
    #get the max purchase date for each customer and create a dataframe with it
    tx_max_purchase = tx_data.groupby('ID')['Date'].max().reset_index()
    tx_max_purchase.columns = ['ID','MaxPurchaseDate']
    #we take our observation point as the max invoice date in our dataset
    #valutiamo la recency come il numero di giorni tra l'ultimo acquisto e oggi
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
    #merge this dataframe to our new user dataframe
    tx_user = pd.merge(tx_user, tx_max_purchase[['ID','Recency']], on='ID')
    #build 4 clusters for recency and add it to dataframe
    kmeans_recency = KMeans(n_clusters=Recency_n_cluster)
    kmeans_recency.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans_recency.predict(tx_user[['Recency']])
    tx_user = order_cluster('RecencyCluster', 'Recency', tx_user, False)

    # st.title('Frequency')
    #numero di acquisti totali fatti da ciascun customers
    tx_frequency = tx_data.groupby('ID')['Date'].count().reset_index()
    tx_frequency.columns = ['ID','Frequency']
    #add this data to our main dataframe
    tx_user = pd.merge(tx_user, tx_frequency, on='ID')
    #k-means
    kmeans_frequency = KMeans(n_clusters=Frequency_n_cluster)
    kmeans_frequency.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans_frequency.predict(tx_user[['Frequency']])
    #order the frequency cluster
    tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

    # st.title('Revenue')
    #calculate revenue for each customer
    tx_revenue = tx_data.groupby('ID')['Revenue'].sum().reset_index()
    #merge it with our main dataframe
    tx_user = pd.merge(tx_user, tx_revenue, on='ID')

    #apply clustering
    kmeans_revenue = KMeans(n_clusters=Revenue_n_cluster)
    kmeans_revenue.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans_revenue.predict(tx_user[['Revenue']])
    #order the cluster numbers
    tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

    # st.header('overall score')
    #calculate overall score and use mean() to see details
    tx_user['RFM_Score'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
    tx_value = tx_user.groupby('RFM_Score')['Recency', 'Frequency', 'Revenue'].mean()
    tx_value['Customer Value (RFM)'] = tx_value.index
    st.write(tx_value)

    #apply segmentation based on the overall score
    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['RFM_Score']>2,'Segment'] = 'Mid-Value'
    tx_user.loc[tx_user['RFM_Score']>4,'Segment'] = 'High-Value'
    return tx_user, kmeans_recency, kmeans_frequency, kmeans_revenue

eda = st.checkbox('Exploratory Data Analysis')

Recency_n_cluster = 4
Frequency_n_cluster = 4
Revenue_n_cluster = 4
Life_time_cluster = int(st.sidebar.text_input('Life time num Cluster', 3))
time_window_pred = int(st.sidebar.text_input('Number of month on which you want to predict the customer lifetime', 6))
time_window_train = int(st.sidebar.text_input('Number of previous month for training', 3))

tx_data_example = pd.read_csv('marketing1.csv')
st.subheader('the application can read data with the following arrangement:')
st.write(tx_data_example.head(20))
st.write('**Date**: Day and time of the purchase')
st.write('**ID**: Customer ID number')
st.write('**Revenue**: income for that specific transaction')

st.write('All the analysis are base on:')
st.write('**Recency**: Number of days from today and the last purchase')
st.write('**Frequency**: Number of purchase per day per customer')
st.write('**Revenue**: Total Revenue per customer')

tx_data_temp = st.file_uploader("Upload dataset", type = ["csv"])

if tx_data_temp:
    tx_data = tx_data_temp
    tx_data = pd.DataFrame(tx_data)
    tx_data[0] = tx_data[0].str.replace('\r\n','')
    tx_data = tx_data[0].str.split(",", expand=True)
    colonne = tx_data.iloc[0].tolist()
    tx_data.columns = colonne
    tx_data = tx_data.drop([0], axis=0)
    tx_data['Date'] = pd.to_datetime(tx_data['Date'])
    tx_data['Revenue'] = tx_data['Revenue'].astype(float)
    tx_data['ID'] = tx_data['ID'].astype(float)
    tx_data['ID'] = tx_data['ID'].astype(int)
    st.header('data collected from the user')
else:
    st.header('data read from the example database')
    tx_data = tx_data_example
    tx_data['Date'] = pd.to_datetime(tx_data['Date'])

#create a dataframe contaning CustomerID and first purchase date
tx_min_purchase = tx_data.groupby('ID')['Date'].min().reset_index()
tx_min_purchase.columns = ['ID','MinPurchaseDate']
#merge first purchase date column to our main dataframe (tx_uk)
tx_data = pd.merge(tx_data, tx_min_purchase, on='ID')
tx_data['UserType'] = 'Existing'
loaction_first_order = tx_data['Date'].dt.to_period('M') == tx_data['MinPurchaseDate'].dt.to_period('M')
tx_data.loc[loaction_first_order, 'UserType'] = 'New'
st.write(tx_data.head())

# st.title('monthly revenue')
# calcolare la revenue totale in ogni mese
tx_revenue = tx_data.set_index('Date').groupby(pd.Grouper(freq='M'))['Revenue'].sum().reset_index()
ds().nuova_fig(1, indice_subplot =121, width =12, height =5)
ds().titoli(titolo="monthly revenue", ytag='revenue')
ds().dati(x = tx_revenue['Date'], y = tx_revenue['Revenue'])

# st.title('monthly active customers')
#creating monthly active customers dataframe by counting unique Customer IDs
tx_monthly_active = tx_data.set_index('Date').groupby(pd.Grouper(freq='M'))['ID'].nunique().reset_index()
ds().nuova_fig(1, indice_subplot =122, width =12, height =5)
ds().titoli(titolo="monthly active customers", ytag='Num Customers')
ds().dati(x = tx_monthly_active['Date'], y = tx_monthly_active['ID'])
if eda: st.pyplot()


# st.title('Average Revenue per Order')
# create a new dataframe for average revenue by taking the mean of it
tx_monthly_order_avg = tx_data.set_index('Date').groupby(pd.Grouper(freq='M'))['Revenue'].mean().reset_index()
ds().nuova_fig(2, indice_subplot =121, width =12, height =5)
ds().titoli(titolo="Average Revenue per Order", ytag='Revenue per Order')
ds().dati(x = tx_monthly_order_avg['Date'], y = tx_monthly_order_avg['Revenue'])

# st.title('New Customer Ratio')
#calculate the Revenue per month for each user type
tx_user_type_revenue_new = tx_data[tx_data['UserType'] == 'New'].set_index('Date').groupby(pd.Grouper(freq='M'))['Revenue'].sum().reset_index()
tx_user_type_revenue_ex = tx_data[tx_data['UserType'] == 'Existing'].set_index('Date').groupby(pd.Grouper(freq='M'))['Revenue'].sum().reset_index()
ds().nuova_fig(2, indice_subplot =122, width =12, height =5)
ds().titoli(titolo="New Customer vs Existing Customer", ytag='Revenue')
ds().dati(x = tx_user_type_revenue_new['Date'], y = tx_user_type_revenue_new['Revenue'], colore='red', descrizione='New')
ds().dati(x = tx_user_type_revenue_ex['Date'], y = tx_user_type_revenue_ex['Revenue'], descrizione='Existing')
ds().legenda()
if eda: st.pyplot()


# st.title('Monthly Retention Rate')
# st.header('percentuale di customers che sono venuti anche il mese precedente')
#identify which users are active by looking at their revenue per month
tx_data_copy = pd.DataFrame()
tx_data_copy = tx_data.copy()
tx_data_copy['Date_month'] = tx_data_copy['Date'].dt.to_period('M')
tx_user_purchase = tx_data_copy.groupby(['ID','Date_month'])['Revenue'].sum().reset_index()

# #create retention matrix with crosstab
tx_retention = pd.crosstab(tx_user_purchase['ID'], tx_user_purchase['Date_month']).reset_index()
# st.write(tx_retention.head(10))

# #create an array of dictionary which keeps Retained & Total User count for each month
months = tx_retention.columns[1:]
retention_data = pd.DataFrame()
for i in range(months.shape[0] - 1):
    InvoiceYearMonth = months[i+1]
    TotalUserCount = tx_retention[months[i+1]].sum()
    RetainedUserCount = tx_retention[(tx_retention[months[i+1]]>0) & (tx_retention[months[i]]>0)][months[i+1]].sum()
    retention_data[months[i]] = [InvoiceYearMonth, TotalUserCount, RetainedUserCount]

retention_data = retention_data.T
retention_data.columns = ['InvoiceYearMonth', 'TotalUserCount', 'RetainedUserCount']
#convert the array to dataframe and calculate Retention Rate
retention_data['RetentionRate'] = retention_data['RetainedUserCount']/retention_data['TotalUserCount']
# st.write(retention_data.head(40))
def parser(x):
    return datetime.strptime(x, '%Y-%m')
retention_data['InvoiceYearMonth'] = retention_data['InvoiceYearMonth'].astype(str).apply(parser)
ds().nuova_fig(3, indice_subplot =121, width =12, height =5)
ds().titoli(titolo="Monthly Retention Rate", ytag='Retention Rate')
ds().dati(x = retention_data['InvoiceYearMonth'], y = retention_data['RetentionRate'])

if eda: st.title('Cohort Based Retention Rate')
#create our retention table again with crosstab() and add firs purchase year month view
tx_retention = pd.merge(tx_retention,tx_min_purchase,on='ID')

new_column_names = [ 'm_' + str(column) for column in tx_retention.columns[:-1]]
new_column_names.append('MinPurchaseYearMonth')
tx_retention.columns = new_column_names
tx_retention['MinPurchaseYearMonth'] = tx_retention['MinPurchaseYearMonth'].dt.to_period('M')

#create the array of Retained users for each cohort monthly
retention_data = pd.DataFrame()
for i in range(len(months)):
    retention_data_month = []
    total_user_count = tx_retention[tx_retention['MinPurchaseYearMonth'] == months[i]]['MinPurchaseYearMonth'].count()
    retention_data_month.append(total_user_count)
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data_month.append(np.nan)
    retention_data_month.append(1)

    for next_month in next_months:
        val = tx_retention[tx_retention['MinPurchaseYearMonth'] == months[i]][tx_retention['m_'+str(next_month)] > 0]['m_'+str(next_month)].sum()/total_user_count
        retention_data_month.append(round(val ,2))
    retention_data[months[i]] = retention_data_month

retention_data = retention_data.T

col_vet = ['TotalUserCount']
for month in months:
    col_vet.append(str(month))
retention_data.columns = [col_vet]
index_retention = retention_data.index.astype(str).to_list()
retention_data = retention_data.reset_index(drop = True)
retention_data.index = index_retention
if eda: st.write(retention_data)

tx_user, _, _, _ = rec_freq_rev(tx_data)

ds().nuova_fig(3, indice_subplot =122, width =12, height =5)
ds().titoli(titolo="Recency", xtag='days from last purchase')
ds().dati(x= tx_user['Recency'], y = 30, scat_plot = 'hist')
if eda: st.pyplot()

ds().nuova_fig(4, indice_subplot =121, width =12, height =5)
ds().titoli(titolo="Frequency", xtag='Total purchase per client')
ds().dati(x= tx_user['Frequency'], y = 300, scat_plot = 'hist')
ds().range_plot(topX = 1000, bottomX = 0)

ds().nuova_fig(4, indice_subplot =122, width =12, height =5)
ds().dati(x= tx_user['Revenue'], y = 600, scat_plot = 'hist')
ds().titoli(titolo="Revenue", xtag='Total Revenue per client')
ds().range_plot(topX = 10000, bottomX = -1000)
if eda: st.pyplot()

ds().nuova_fig(5, indice_subplot =311, width =6, height =10)
ds().titoli(titolo='', xtag="frequency", ytag='Revenue')
ds().dati(x = tx_user[tx_user['Segment'] == 'High-Value']['Frequency'], y = tx_user[tx_user['Segment'] == 'High-Value']['Revenue'], scat_plot = 'scat', colore='red', descrizione='High-Value')
ds().dati(x = tx_user[tx_user['Segment'] == 'Mid-Value']['Frequency'], y = tx_user[tx_user['Segment'] == 'Mid-Value']['Revenue'], scat_plot = 'scat', colore='green', descrizione='Mid-Value')
ds().dati(x = tx_user[tx_user['Segment'] == 'Low-Value']['Frequency'], y = tx_user[tx_user['Segment'] == 'Low-Value']['Revenue'], scat_plot = 'scat', colore='blue', descrizione='Low-Value')
ds().range_plot(bottomX = -10, topX = 2000, bottomY = -1000, topY = 40_000)
ds().legenda()

ds().nuova_fig(5, indice_subplot =312, width =6, height =10)
ds().titoli(titolo='', xtag="Recency", ytag='Revenue')
ds().dati(x = tx_user[tx_user['Segment'] == 'High-Value']['Recency'], y = tx_user[tx_user['Segment'] == 'High-Value']['Revenue'], scat_plot = 'scat', colore='red', descrizione='High-Value')
ds().dati(x = tx_user[tx_user['Segment'] == 'Mid-Value']['Recency'], y = tx_user[tx_user['Segment'] == 'Mid-Value']['Revenue'], scat_plot = 'scat', colore='green', descrizione='Mid-Value')
ds().dati(x = tx_user[tx_user['Segment'] == 'Low-Value']['Recency'], y = tx_user[tx_user['Segment'] == 'Low-Value']['Revenue'], scat_plot = 'scat', colore='blue', descrizione='Low-Value')
ds().range_plot(bottomX = -10, topX = 400, bottomY = -1000, topY = 40_000)
ds().legenda()

ds().nuova_fig(5, indice_subplot =313, width =6, height =10)
ds().titoli(titolo='', xtag="Recency", ytag='Frequency')
ds().dati(x = tx_user[tx_user['Segment'] == 'High-Value']['Recency'], y = tx_user[tx_user['Segment'] == 'High-Value']['Frequency'], scat_plot = 'scat', colore='red', descrizione='High-Value')
ds().dati(x = tx_user[tx_user['Segment'] == 'Mid-Value']['Recency'], y = tx_user[tx_user['Segment'] == 'Mid-Value']['Frequency'], scat_plot = 'scat', colore='green', descrizione='Mid-Value')
ds().dati(x = tx_user[tx_user['Segment'] == 'Low-Value']['Recency'], y = tx_user[tx_user['Segment'] == 'Low-Value']['Frequency'], scat_plot = 'scat', colore='blue', descrizione='Low-Value')
ds().range_plot(bottomX = -10, topX = 400, bottomY = -10, topY = 2000)
ds().legenda()
ds().aggiusta_la_finestra()
if eda: st.pyplot()

st.title('recency, frequency and revenue over 3 month')
#create 3m and 6m dataframes
#6 mesi database serve per il training che verra usato per predirre il lifetime sui 6 mesi successivi
month = tx_data.set_index('Date').groupby(pd.Grouper(freq='M'))['Revenue'].count().reset_index()['Date'].tolist()
tx_3m = tx_data[(tx_data['Date'] < month[-time_window_pred]) & (tx_data['Date'] >= month[-(time_window_pred+time_window_train)])].reset_index(drop=True)
tx_6m = tx_data[(tx_data['Date'] >= month[-time_window_pred]) & (tx_data['Date'] < month[-1])].reset_index(drop=True)

tx_user_3m, kmeans_recency, kmeans_frequency, kmeans_revenue = rec_freq_rev(tx_3m)

tx_user_6m = tx_6m.groupby('ID')['Revenue'].sum().reset_index()
tx_user_6m.columns = ['ID','m6_Revenue_LFV']

tx_merge = pd.merge(tx_user_3m, tx_user_6m, on='ID', how='left')
tx_merge = tx_merge.dropna()
#remove outliers
st.write('outliers are remove from the dataset')
tx_merge = tx_merge[tx_merge['m6_Revenue_LFV']<tx_merge['m6_Revenue_LFV'].quantile(0.99)]

ds().nuova_fig(6, width = 12)
ds().titoli(titolo='RFM vs LFV without outliers', xtag="RFM_Score (Customer Value)", ytag='m6_Revenue_LFV (Life Time)')
ds().dati(x = tx_merge[tx_merge['Segment'] == 'High-Value']['RFM_Score'], y = tx_merge[tx_merge['Segment'] == 'High-Value']['m6_Revenue_LFV'], scat_plot = 'scat', colore='red', descrizione= 'High-Value')
ds().dati(x = tx_merge[tx_merge['Segment'] == 'Mid-Value']['RFM_Score'], y = tx_merge[tx_merge['Segment'] == 'Mid-Value']['m6_Revenue_LFV'], scat_plot = 'scat', colore='green', descrizione= 'Mid-Value')
ds().dati(x = tx_merge[tx_merge['Segment'] == 'Low-Value']['RFM_Score'], y = tx_merge[tx_merge['Segment'] == 'Low-Value']['m6_Revenue_LFV'], scat_plot = 'scat', colore='blue', descrizione= 'Low-Value')
ds().range_plot(topY = 30_000, bottomY = 0)
ds().legenda()
ds().aggiusta_la_finestra()
st.pyplot()

#creating 3 clusters
kmeans_lifetime = KMeans(n_clusters=Life_time_cluster)
kmeans_lifetime.fit(tx_merge[['m6_Revenue_LFV']])
tx_merge['LTVCluster'] = kmeans_lifetime.predict(tx_merge[['m6_Revenue_LFV']])
#order cluster number based on LTV
tx_merge = order_cluster('LTVCluster', 'm6_Revenue_LFV', tx_merge, True)
# st.write(tx_merge.head())
LTV_classes = tx_merge['LTVCluster'].unique().tolist()
st.subheader('life time classes')
life_time_matrix = tx_merge.groupby('LTVCluster')['Recency','Frequency','Revenue', 'm6_Revenue_LFV'].mean()
st.write(life_time_matrix)

#convert categorical columns to numerical
tx_class = tx_merge.copy()
le = LabelEncoder()
tx_class['Segment'] = le.fit_transform(tx_class['Segment'])
# st.write(tx_class.head())

# tx_class = tx_class.drop(['Recency', 'Frequency', 'Revenue'], axis = 1)
# tx_class = tx_class.drop(['RecencyCluster', 'FrequencyCluster', 'RevenueCluster'], axis = 1)

#calculate and show correlations
cor1, cor2 = marketing().correlation_matrix(tx_class, 'LTVCluster')

#create X and y, X will be feature set and y is the label - LTV
X = tx_class.drop(['LTVCluster','m6_Revenue_LFV', 'ID'],axis=1)
y = tx_class['LTVCluster']

#split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# #XGBoost Multiclassification Model
ltv_xgb_model = xgb.XGBClassifier(n_jobs = -1).fit(X_train, y_train)
train_score = ltv_xgb_model.score(X_train, y_train)
test_score = ltv_xgb_model.score(X_test, y_test)
y_pred = ltv_xgb_model.predict(X_test)
df_pred = pd.DataFrame()
df_pred['xgboost'] = y_pred
st.title('Model Confusion Matrix on Test set')
marketing().score_accuracy_recall(df_pred, y_test, verbose = 1)
st.header('overall accuracy on the test set: ' + str(round(ltv_xgb_model.score(X_test[X_train.columns], y_test)*100, 2))+"%")

# st.write(X_test.head())

recency = st.text_input('Recency')
frequency = st.text_input('Frequency')
revenue = st.text_input('Revenue')

if recency and frequency and revenue:
    df_single = pd.DataFrame()
    recency = int(recency)
    frequency = int(frequency)
    revenue = float(revenue)

    RecencyCluster = kmeans_recency.predict(np.array([recency]).reshape(1,-1))
    FrequencyCluster = kmeans_frequency.predict(np.array([frequency]).reshape(1,-1))
    RevenueCluster = kmeans_revenue.predict(np.array([revenue]).reshape(1,-1))

    df_single['Recency'] = [recency]
    df_single['RecencyCluster'] = RecencyCluster

    df_single['Frequency'] = [frequency]
    df_single['FrequencyCluster'] = FrequencyCluster

    df_single['Revenue'] = [revenue]
    df_single['RevenueCluster'] = RevenueCluster

    df_single['RFM_Score'] = RecencyCluster + FrequencyCluster + RevenueCluster

    df_single['Segment'] = ['Low-Value']
    df_single.loc[df_single['RFM_Score']>2,'Segment'] = 'Mid-Value'
    df_single.loc[df_single['RFM_Score']>4,'Segment'] = 'High-Value'
    st.write('this is the shape of your data:')
    st.write(df_single)
    df_single['Segment'] = le.transform(df_single['Segment'])
    y_pred = ltv_xgb_model.predict(df_single)
    st.write('the lifetime is divided in **'+str(len(LTV_classes))+'** classes')
    st.write('**'+str(LTV_classes[0])+'** is the shortest life time and **' + str(LTV_classes[-1]) + '** is the logest life time')
    st.write('your sample for the next **'+str(time_window_pred)+'** months belogs to class **'+str(y_pred[0])+'** which correspond to an average revenue of **'+str(round(life_time_matrix.loc[y_pred[0], 'm6_Revenue_LFV'],2))+'**')
