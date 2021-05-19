#!/usr/bin/env python
# coding: utf-8

# K-means clustering will be used for both parts of the project: Customer segmentation using RFM.

# # Setup
# import necessary librairies
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots
from IPython import get_ipython

import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide')
st.footer("Developed By [Yasmina Tabbara](https://www.linkedin.com/in/yasmina-tabbara/) as part of a Data-Driven Digital Marketing Course")

#ccreate columns to center pic
col1, col2, col3 = st.beta_columns([2,5,2])
col2.image('https://d35fo82fjcw0y8.cloudfront.net/2018/08/09094017/psychographic_segmentation_header-11-e1551338916467.jpg')

# Create a sidebar
st.sidebar.title("Customer Segmentation")
st.sidebar.subheader('Data Upload')
# create an upload button for the dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset here", type = ['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
rad = st.sidebar.radio(' ',['Introduction', "Data Exploration", 'RFM Analysis', "K-Means Clustering", "Customer Segment Search", "Cluster Calculator"])
nb = st.sidebar.slider('Slide to choose your number of clusters', 2, 10)

#introduction section
if rad == 'Introduction':
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>Customer Segmentation using RFM and K-Means Clustering</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: Black;'>by Yasmina Tabbara</h3>", unsafe_allow_html=True)
    st.write("Customer segmentation is a key aspect of Marketing, allowing the businesses to better understand the behavior of their customers and targeting them more efficiently. Traditional methods include certain segmentation bases such as Geographical, Demographic, or Behavioral. One of the most famous methods is by using RFM which track's customers' buying behavior including the recency, frequency and monetary value of their purchases. However, RFM scores are usually pre-determined and can take a long time to calculate and apply. Here is where Machine Learning comes in and makes things much easier. By using unsupervised ML models, we can automatically detect different clusters in our customers based on their transactions.")
    st.subheader("Start by uploading your dataset in the sidebar!")
    st.write('Once it is done, you can click on the button below to take a look at the first 10 rows of the dataset.')
    if st.button("Click me"):
        st.write(data.head(10))

#transform order date to correct format and create new column with year
data['OrderDate'] = pd.to_datetime(data['Order Date']).dt.date
data = data.drop(columns='Order Date')
data['Year'] = pd.DatetimeIndex(data['OrderDate']).year

#new section: data exploration
if rad == 'Data Exploration':
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>Data Exploration</h1>", unsafe_allow_html=True)
    st.subheader("This dashbaord aims to provide some insights around the Sales of the store.")
    st.write("Start by selecting the Country and Segment you'd like to analyze in the Navigation Bar on the left.")

    # filter by country
    filt_cn = data['Country'].unique()
    slct_cn = st.sidebar.selectbox('Select the Country:', filt_cn)
    filtered_data = data.loc[data['Country'] == slct_cn]

    #filter by segment
    filt_sg = data['Segment'].unique()
    slct_sg = st.sidebar.selectbox('Select the Segment:', filt_sg)
    filtered_data = filtered_data.loc[filtered_data['Segment'] == slct_sg]

    st.header("Sales Analysis for the {} Segment, in {}".format(slct_sg, slct_cn))
     
    #map to visualize country transactions
    fig = px.scatter_geo(filtered_data, lat="Latitude (generated)", lon='Longitude (generated)', color="City",
                     hover_name="City", size='Sales',
                     animation_frame="Year",
                     width = 1500,
                     height = 800,
                     projection="natural earth", color_continuous_scale=px.colors.sequential.Agsunset, title = 'Sales in different cities of')
    fig.update_layout(title="The distribution of sales in the cities of {} for the {} segment".format(slct_cn, slct_sg))
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig)
    

    col1, col2 = st.beta_columns((1,1))

    #sales across time
    fig = px.histogram(filtered_data, x='OrderDate', y='Sales', histfunc='sum', height = 500, width = 800, color_discrete_sequence=px.colors.sequential.Agsunset)
    fig.update_layout(title="Evolution of Sales in {} for the {} segment".format(slct_cn, slct_sg))
    fig.update_traces(xbins_size="M1")
    fig.update_xaxes(showgrid=True, ticklabelmode="period", dtick="M1")
    fig.update_layout(template='plotly_white',bargap=0.1)
    col1.plotly_chart(fig)
    col1.subheader(" ")

    #pie chart for sales across cats
    fig = px.pie(filtered_data, values='Sales', names='Category',
    color_discrete_sequence=px.colors.sequential.Agsunset, height = 520, width = 500)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title="Sales Across Categories in {} for the {} segment".format(slct_cn, slct_sg))
    fig.update_layout(template='plotly_white', showlegend=False)
    col2.plotly_chart(fig)
    col2.subheader(' ')

    # sub-category and profit
    fig = go.Figure()
    fig.add_trace(go.Bar(x=filtered_data['Sub-Category'], y=filtered_data.Sales,
                    marker_color='purple',
                    name='Sales'))
    fig.add_trace(go.Bar(x=filtered_data['Sub-Category'], y=filtered_data.Profit,
                    marker_color='DarkSlateBlue',
                    name='Profits'
                    ))
    fig.update_layout(template='plotly_white',width=750, height =710)
    fig.update_layout(title="Sales and Profits of Sub-Categories in {} for the {} segment".format(slct_cn, slct_sg))
    col1.plotly_chart(fig)

    # relationship between discount and sales
    fig = px.scatter(filtered_data, x='Profit', y='Discount', color_discrete_sequence=px.colors.sequential.Agsunset,
    height = 700, width = 750)
    fig.update_layout(title="Relationship Between Profits and Discounts in {} for the {} segment".format(slct_cn, slct_sg))
    fig.update_layout(template='plotly_white')
    col2.plotly_chart(fig)


# create subset with selected columns
columns = ['Customer ID', 'OrderDate', 'Sales']
data_rfm = data[columns]

# create new dataframe with customer ids
rfm_df = pd.DataFrame(data_rfm['Customer ID'].unique())
rfm_df.columns = ['CustomerID']

# creating new values of RFM
## Recency
rfm_df_recent = data.groupby('Customer ID').OrderDate.max().reset_index()
# create a dataframe with these two columns
rfm_df_recent.columns = ['CustomerID', 'MostRecentDate']
rfm_df_recent['Recency'] = (rfm_df_recent['MostRecentDate'].max() - rfm_df_recent['MostRecentDate']).dt.days
# merge this dataframe with the one related to customers
rfm_df = pd.merge(rfm_df, rfm_df_recent[['CustomerID','Recency']], on='CustomerID')

#Frequency: similarly to what we did for recency, we should get order counts for each customer and create a dataframe with it
rfm_df_frequent = data.groupby('Customer ID').OrderDate.nunique().reset_index()
rfm_df_frequent.columns = ['CustomerID', 'Frequency']
# add new column to our rfm dataframe
rfm_df = pd.merge(rfm_df, rfm_df_frequent, on='CustomerID')

# Monetary
rfm_df_monetary = data.groupby('Customer ID').Sales.sum().reset_index()
rfm_df_monetary.columns = ['CustomerID', 'Monetary']
# add new column to our rfm dataframe
rfm_df = pd.merge(rfm_df, rfm_df_monetary, on='CustomerID')

# Rounding values
rfm_df = rfm_df.round(decimals=0)

#new section about analysis
if rad == 'RFM Analysis':
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>RFM Analysis</h1>", unsafe_allow_html=True)
    st.subheader("Now, we must calculate each element of the RFM. But before doing that, we must create a new dataframe that contains each unique customer id and then we can add the relevant values.")

    # Recency
    st.write("To calculate Recency, we must first get the date of the most recent purchase.")
    st.write("We will take our reference point as the max invoice date in our dataset which represents the most recent date, and our recency will be based on days.")
    # Frequency
    st.write("For Frequency, we will count the distinct number of times that each customer has placed an order.")

    ## Monetary
    st.write("And finally, for Monetary Value, we will sum up the Sales of each customer to find how much he has spent in total.")

    # Rounding values to make it easier for Clustering
    st.write("We will also round our values to the nearest integer in order to simplify the data and normalize it.")

    st.subheader('And this is what our dataframe looks like:')
    st.table(rfm_df.iloc[0:10])

    st.write("Would you like to take a look at the distributions of the RFM elements?")
    if st.button('Yes!'):
        col1, col2, col3 = st.beta_columns((1,1,1))
        # Recency
        fig = px.histogram(rfm_df, x='Recency', color_discrete_sequence=px.colors.sequential.Agsunset)
        fig.update_layout(title="Distribution of Recency", template='plotly_white', width=550)
        col1.plotly_chart(fig)

        # Frequency
        fig = px.histogram(rfm_df, x='Frequency', color_discrete_sequence=['darkmagenta'])
        fig.update_layout(title="Distribution of Frequency", template='plotly_white', width=550)
        col2.plotly_chart(fig)

        # Monetary
        fig = px.histogram(rfm_df, x='Monetary', color_discrete_sequence=['plum'])
        fig.update_layout(title="Distribution of Monetary Value", template='plotly_white', width=550)
        col3.plotly_chart(fig)


    st.subheader("Before we can proceed with applying K-Means clustering, our data has to be standardized and outliers should be removed as k-means is sensitive to outliers! Thus, removing them will help us get a better picture and representation of our customers.")
    st.write("Let us check for outliers in each of the RFM elements.")
    if st.checkbox('Check this box to remove outliers'):
        # # # Removing Outliers
        # # Recency
        # calculate interquartile range
        recency_q25, recency_q75 = np.percentile(rfm_df['Recency'], 25), np.percentile(rfm_df['Recency'], 75)
        recency_iqr = recency_q75 - recency_q25
        print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (recency_q25, recency_q75, recency_iqr))
        # calculate the outlier cutoff
        recency_iqr_cut_off = recency_iqr * 1.5
        recency_iqr_lower, recency_iqr_upper = recency_q25 - recency_iqr_cut_off, recency_q75 + recency_iqr_cut_off
        # Remove outliers from dataset
        rfm_df = rfm_df[(rfm_df.Recency<recency_iqr_upper) & (rfm_df.Recency>recency_iqr_lower)]

        # # Frequency
        # calculate interquartile range
        frequency_q25, frequency_q75 = np.percentile(rfm_df['Frequency'], 25), np.percentile(rfm_df['Frequency'], 75)
        frequency_iqr = frequency_q75 - frequency_q25
        print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (frequency_q25, frequency_q75, frequency_iqr))
        # calculate the outlier cutoff
        frequency_iqr_cut_off = recency_iqr * 1.5
        frequency_iqr_lower, frequency_iqr_upper = frequency_q25 - frequency_iqr_cut_off, frequency_q75 + frequency_iqr_cut_off
        # Remove outliers from dataset
        rfm_df = rfm_df[(rfm_df.Frequency<frequency_iqr_upper) & (rfm_df.Frequency>frequency_iqr_lower)]

        # # Monetary Value
        # calculate interquartile range
        monetary_q25, monetary_q75 = np.percentile(rfm_df['Monetary'], 25), np.percentile(rfm_df['Monetary'], 75)
        monetary_iqr = monetary_q75 - monetary_q25
        print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (monetary_q25, monetary_q75, monetary_iqr))
        # calculate the outlier cutoff
        monetary_iqr_cut_off = monetary_iqr * 1.5
        monetary_iqr_lower, monetary_iqr_upper = monetary_q25 - monetary_iqr_cut_off, monetary_q75 + monetary_iqr_cut_off
        # Remove outliers from dataset
        rfm_df = rfm_df[(rfm_df.Monetary<monetary_iqr_upper) & (rfm_df.Monetary>monetary_iqr_lower)]

    col1, col2, col3 = st.beta_columns((1,1,1))

    # recency
    fig = px.box(rfm_df, y="Recency", color_discrete_sequence=px.colors.sequential.Agsunset)
    fig.update_layout(title="Recency Boxplot", template='plotly_white', width=550)
    col1.plotly_chart(fig)

    # frequency
    fig = px.box(rfm_df, y="Frequency", color_discrete_sequence=['darkmagenta'])
    fig.update_layout(title="Frequency Boxplot", template='plotly_white', width=550)
    col2.plotly_chart(fig)

    # monetary
    fig = px.box(rfm_df, y="Monetary", color_discrete_sequence=['plum'])
    fig.update_layout(title="Monetary Value Boxplot", template='plotly_white', width=550)
    col3.plotly_chart(fig)



# # # Removing Outliers
# # Recency
# calculate interquartile range
recency_q25, recency_q75 = np.percentile(rfm_df['Recency'], 25), np.percentile(rfm_df['Recency'], 75)
recency_iqr = recency_q75 - recency_q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (recency_q25, recency_q75, recency_iqr))
# calculate the outlier cutoff
recency_iqr_cut_off = recency_iqr * 1.5
recency_iqr_lower, recency_iqr_upper = recency_q25 - recency_iqr_cut_off, recency_q75 + recency_iqr_cut_off
# Remove outliers from dataset
rfm_df = rfm_df[(rfm_df.Recency<recency_iqr_upper) & (rfm_df.Recency>recency_iqr_lower)]

# # Frequency
# calculate interquartile range
frequency_q25, frequency_q75 = np.percentile(rfm_df['Frequency'], 25), np.percentile(rfm_df['Frequency'], 75)
frequency_iqr = frequency_q75 - frequency_q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (frequency_q25, frequency_q75, frequency_iqr))
# calculate the outlier cutoff
frequency_iqr_cut_off = recency_iqr * 1.5
frequency_iqr_lower, frequency_iqr_upper = frequency_q25 - frequency_iqr_cut_off, frequency_q75 + frequency_iqr_cut_off
# Remove outliers from dataset
rfm_df = rfm_df[(rfm_df.Frequency<frequency_iqr_upper) & (rfm_df.Frequency>frequency_iqr_lower)]

# # Monetary Value
# calculate interquartile range
monetary_q25, monetary_q75 = np.percentile(rfm_df['Monetary'], 25), np.percentile(rfm_df['Monetary'], 75)
monetary_iqr = monetary_q75 - monetary_q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (monetary_q25, monetary_q75, monetary_iqr))
# calculate the outlier cutoff
monetary_iqr_cut_off = monetary_iqr * 1.5
monetary_iqr_lower, monetary_iqr_upper = monetary_q25 - monetary_iqr_cut_off, monetary_q75 + monetary_iqr_cut_off
# Remove outliers from dataset
rfm_df = rfm_df[(rfm_df.Monetary<monetary_iqr_upper) & (rfm_df.Monetary>monetary_iqr_lower)]


# # # Data Scaling
# Data scaling should be applied because it will minimize the errors in clustering
rfm = rfm_df.drop(columns='CustomerID')
ss = StandardScaler()
rfm = ss.fit_transform(rfm)


if rad == 'K-Means Clustering':
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>K-Means Clustering for Segmentation</h1>", unsafe_allow_html=True)
    st.subheader('For this task, we will be using an unsupervised Machine Learning algorithm which is K-Means which identifies k number of centroids, and then allocates every data point to the nearest cluster (based on similarities), while keeping the centroids as small as possible.')
    # # # Applying K-Means Clustering
    #  # Choosing Optimal Number of Clusters
    st.write('Using the elbow method (which is picking the elbow of the curve as the number of clusters to use), we will try to find the optimal number of clusters for our data.')
    # generate elbow plot
    sse={}
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(rfm)
        sse[k] = kmeans.inertia_
    col1, col2 = st.beta_columns(2)
    fig = go.Figure(data=go.Scatter(x=list(sse.keys()), y=list(sse.values()), marker=dict(color='Purple')))
    fig.update_xaxes(title_text='Number of Clusters')
    fig.update_yaxes(title_text='Inertia')
    fig.update_layout(height = 600, width = 700, title_text='Elbow Method for Optimal K', template='plotly_white')
    col1.plotly_chart(fig)

    x = [round(num, 1) for num in list(sse.keys())]
    y = [round(num, 1) for num in list(sse.values())]

    #create figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Number of Clusters', 'Inertia'],
                    line_color='black',
                    fill_color='plum',
                    font = dict(color = 'white', size = 15)),
        cells=dict(values=[x, y],
                    line_color='black',
                    fill_color='white',
                    font = dict(size = 14),
                    height = 25))
            ])
    fig.update_layout(height=600, width=700)
    col2.plotly_chart(fig)


# ### Visualizing silhouette analysis
def silhouette(X, n_clusters):
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=1,
                              print_grid=False)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1
    fig['layout']['xaxis1'].update(title='The silhouette coefficient values',
                                   range=[-0.1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    fig['layout']['yaxis1'].update(title='Cluster label',
                                   showticklabels=False,
                                   range=[0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 42 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10

    colors = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        values = float(i) / n_clusters
        filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                 x=ith_cluster_silhouette_values,
                                 mode='lines',
                                 showlegend=False,
                                 line=dict(width=0.5, color=colors[i]),
                                 fill='tozerox')

        # add a vertical line to show the silhouette score
        line = fig.add_vline(x=silhouette_avg,
                          line_width=1,
                          line_dash="dash",
                          line_color="red")

        fig.update_layout(template='plotly_white', height = 650, width = 700)
        fig.append_trace(filled_area, 1, 1)
        fig['layout'].update(title="The silhouette plot of the clustered data with n_clusters = %d" % n_clusters)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    col1.plotly_chart(fig)

def clusters(X, range_n_clusters):

    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 42 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)


        # Plot showing the actual clusters formed
        values = cluster_labels.astype(float) / n_clusters
        fig = go.Figure(data=[go.Scatter3d(x=X[:, 0],
                              y=X[:, 1],
                              z=X[:, 2],
                              mode='markers',
                              marker=dict(color = values, colorscale = px.colors.sequential.Purpor,
                                    opacity = 0.8)
                                    )])
        Scene = dict(xaxis = dict(title  = 'Recency -->'),yaxis = dict(title  = 'Frequency -->'),zaxis = dict(title  = 'Monetary Value -->'))

        fig.update_layout(scene = Scene, height = 650, width = 700, title_text='The visualization of the clustered data.', template='plotly_white')

        fig['layout'].update(title="The visualization of the clustered data with n_clusters = %d" % n_clusters)

    col2.plotly_chart(fig)

if rad == 'K-Means Clustering':
    st.subheader('Let us choose a value for the number of clusters to perform the silhouette analysis.')
    n = st.slider('Slide to analyze', 2, 10)
    col1, col2 = st.beta_columns(2)
    silhouette(rfm, n)
    clusters(rfm, range(2, n+1))
    # # Find Segments
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(rfm)
    clusters = kmeans.predict(rfm)
    rfm_df['Cluster'] = clusters

    cluster_counts = rfm_df['Cluster'].value_counts()
    cluster_counts = cluster_counts.rename('Count')

    for i in range(len(cluster_counts)):
        st.write('Cluster number', cluster_counts.index[i], 'contains', cluster_counts.iloc[i], 'customers')

    st.header('RFM Distribution for n_clusters = %d' % n)
    col1, col2, col3 = st.beta_columns((1,1,1))
    #create a table with the average values
    recency = rfm_df.groupby('Cluster')['Recency'].mean().round(decimals=0)
    frequency = rfm_df.groupby('Cluster')['Frequency'].mean().round(decimals=0)
    monetary = rfm_df.groupby('Cluster')['Monetary'].mean().round(decimals=0)
    segments = pd.DataFrame([recency, frequency, monetary, cluster_counts]).T

    #create bar charts for each of the RFM elements to check distribution
    #recency
    fig1 = px.bar(segments, x=segments.index, y=recency, title="Recency",
        labels={
        'index': 'Cluster',
        'y': 'Average'},
        color_discrete_sequence=px.colors.sequential.Agsunset,
        width = 550)
    fig1.update_layout(template='plotly_white')
    col1.plotly_chart(fig1)

    #frequency
    fig2 = px.bar(segments, x=segments.index, y=frequency, title="Frequency",
        labels={
        'index': 'Cluster',
        'y': 'Average'},
        color_discrete_sequence=['darkmagenta'],
        width = 550)
    fig2.update_layout(template='plotly_white')
    col2.plotly_chart(fig2)

    #monetary
    fig3 = px.bar(segments, x=segments.index, y=monetary, title="Monetary Value",
        labels={
        'index': 'Cluster',
        'y': 'Average'},
        color_discrete_sequence=['plum'],
        width = 550)
    fig3.update_layout(template='plotly_white')
    col3.plotly_chart(fig3)

    # Recency Details Table
    rec = rfm_df.groupby('Cluster')['Recency'].describe().T
    col1.table(rec)

    #Frequency
    freq = rfm_df.groupby('Cluster')['Frequency'].describe().T
    col2.table(freq)

    #monetary
    mon = rfm_df.groupby('Cluster')['Monetary'].describe().T
    col3.table(mon)

#perform kmeans on data
kmeans = KMeans(n_clusters=int(nb), random_state=42)
kmeans.fit(rfm)
clusters = kmeans.predict(rfm)
rfm_df['Cluster'] = clusters

if rad == 'Customer Segment Search':
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>Customer Segment Search</h1>", unsafe_allow_html=True)
    st.subheader("In this final section, you can look up the segment of any of our customers and get their RFM details by first choosing the number of clusters you'd like in the sidebar and inserting their ID.")
    st.write("A quick look at what our dataframe looks like:")
    st.table(rfm_df.head())
    #create selectbox with values of ids available
    ids = rfm_df.CustomerID.unique()
    id = st.selectbox('Choose the ID here:', ids)
    if st.button('Click here to get their RFM and transactions details'):
        info = rfm_df.loc[rfm_df["CustomerID"] == id]
        #create dataframe with all of this cutomer's orders
        st.write("Here are all of their transactions:")
        transac = data.loc[data['Customer ID'] == id]
        st.dataframe(transac)
        col1, col2, col3 = st.beta_columns((1,1,1))

        #some plots and figures
        fig = px.histogram(transac, x='OrderDate', y='Sales', height = 500, width = 500, color_discrete_sequence=px.colors.sequential.Agsunset)
        fig.update_layout(title='Sales distribution for customer %s' % id)
        fig.update_layout(template='plotly_white')
        col1.plotly_chart(fig)

        fig = px.pie(transac, values='Sales', names='Category',
        color_discrete_sequence=px.colors.sequential.Agsunset, height = 520, width = 500)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Sales Across Categories for customer %s" % id)
        fig.update_layout(template='plotly_white', showlegend=False)
        col2.plotly_chart(fig)
        col2.subheader(' ')

        #get average spending and biggest order
        avg_sales = transac.Sales.sum() / len(transac)
        max_sale = transac.Sales.max()
        first_purchase = transac.OrderDate.min()
        last_purchase = transac.OrderDate.max()
        profit = transac.Profit.sum()
        cluster = info.Cluster.values
        recency = info.Recency.values
        frequency = info.Frequency.values
        monetary = info.Monetary.values

        col3.subheader('Some insights:')
        col3.write('-This customer belongs to cluster number %d.' % cluster)
        col3.write('-Throughout his time with us, he made %d purchases.' % frequency)
        col3.write('-He spent in average $ %d in our store.' % avg_sales)
        col3.write('-His biggest transaction was for $ %d.' % max_sale)
        col3.write('-His first order was made on %s.' % first_purchase)
        col3.write('-His last order was made on %s' % last_purchase)
        col3.write('which makes it %d days ago.' % recency)
        col3.write('-In total, this customer has generated $ %d in profits for us.' % profit)

        #map to plot his orders
        fig = px.scatter_geo(transac, lat="Latitude (generated)", lon='Longitude (generated)', color="City",
                         hover_name="City", size='Sales',
                         animation_frame="Year",
                         width = 1500,
                         height = 800,
                         projection="natural earth", color_continuous_scale=px.colors.sequential.Agsunset, title = 'Sales in different cities of')
        fig.update_layout(title="The distribution of sales for customer %s" % id)
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig)

if rad == "Cluster Calculator":
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>Cluster Calculator</h1>", unsafe_allow_html=True)
    st.header('Now you can try inputting some values for RFM and see which cluster this imaginary customer can be apart of')
    st.subheader("To be able to do that, we will consider that any customer whose RFM values fall between the IQR of the Cluster average values, can be part of that cluster.")
    unique_clusters = rfm_df['Cluster'].unique()
    col1, col2, col3 = st.beta_columns((1,1,1))
    r= col1.number_input('Add Recency')
    f= col2.number_input('Add Frequency')
    m= col3.number_input('Add Monetary Value')
    c= ' '

    #Recency
    rec_q1 = rfm_df.groupby('Cluster')['Recency'].quantile(0.25)
    rec_q3 = rfm_df.groupby('Cluster')['Recency'].quantile(0.75)

    #Frequency
    freq_q1 = rfm_df.groupby('Cluster')['Frequency'].quantile(0.25)
    freq_q3 = rfm_df.groupby('Cluster')['Frequency'].quantile(0.75)

    #monetary
    monetary_q1 = rfm_df.groupby('Cluster')['Monetary'].quantile(0.25)
    monetary_q3 = rfm_df.groupby('Cluster')['Monetary'].quantile(0.25)

    #iterate for each cluster to see if it fits
    if st.button("Click here to calculate!"):
        for n in range(len(unique_clusters)):
            if (rec_q1[n] <= r <= rec_q3[n]) and (freq_q1[n] <= f <= freq_q3[n]) and (monetary_q1[n] <= m <= monetary_q3[n]):
                c = n
                st.ballons()
                st.success('Congratulations! This customer can be added to cluster number $d' % c)
            else:
                st.error('Try again! This customer does not fit in cluster %d' % n)
