import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
#from dataprep.eda import plot_correlation
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import requests
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from sklearn import preprocessing
from itertools import groupby
from scipy import stats
import random
import math

# Web App Title
st.markdown('''
# **The Customer Segmentation RFM model using K-Means algorithms App**
---
''')

# Upload csv data
with st.sidebar.header('Upload your csv data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input csv file", type=["csv"])

# Pandas Profiling Report
if uploaded_file is not None:
    #@st.cache(allow_output_mutation=True)
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        csv['InvoiceDate'] = csv.InvoiceDate.astype('datetime64[s]') #change type object to datetime
        return csv
    df = load_csv()

    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
   
    #st.header('**Checking Dataframe**')
    #df_shape = df.shape
    #st.write('Shape : ', df_shape)
    #column data type
    ##df_data_type = df.dtypes
    ##st.header('Data Types')
    ##st.write(df_data_type)
    #Number of NaN values per column:
    ##df_Number_NaN = df.isnull().sum()
    ##st.write('Number of NaN values per column : ',df_Number_NaN)
    #st.write(order)#
    #st.write('---')
    #st.header('**Statistics**')
    #df_describe = df.describe().round(2)
    #st.write(df_describe)


    st.header('**Group by Invoice No.**')
    df['Price'] = df['Quantity'] * df['UnitPrice']
    df_order = df.groupby(['InvoiceNo','InvoiceDate','CustomerID']).agg({'Price':lambda x:x.sum()}).reset_index()
    st.write(df_order.head())
    st.write(df_order.describe())
    NOW = df_order['InvoiceDate'].max() + timedelta(days=1)
    period = 365
    df_order['Recency'] = df_order['InvoiceDate'].apply(lambda x:(NOW - x).days)
    aggr = {'Recency':lambda x:x.min(), #the number of days since last order (Recency)
    'InvoiceDate':lambda x:len([d for d in x if d >= NOW - timedelta(days=period)]),} # the total number of order in the last period (Frequency)
    rfm = df_order.groupby('CustomerID').agg(aggr).reset_index()
    rfm.rename(columns={'InvoiceDate':'Frequency'},inplace=True)
    rfm['Monetary'] = rfm['CustomerID'].apply(lambda x:df_order[(df_order['CustomerID']==x) & (df_order['InvoiceDate'] >= NOW - timedelta(days=period))]['Price'].sum())
    st.header('**RFM Table**')
    st.write(rfm.head())
    st.header('**RFM Statistics**')
    st.write(rfm.describe())

    #Assige score to RFM
    quintiles = rfm[['Recency','Frequency','Monetary']].quantile([.2, .4, .6, .8]).to_dict() #ควอนไทล์แจกแจง
    def r_score(x):
        if x <= quintiles['Recency'][.2]:
            return 5
        elif x <= quintiles['Recency'][.4]:
            return 4
        elif x <= quintiles['Recency'][.6]:
            return 3
        elif x <= quintiles['Recency'][.8]:
            return 2
        else:
            return 1
    def fm_score(x, c):
        if x <= quintiles[c][.2]:
            return 1
        if x <= quintiles[c][.4]:
            return 2
        if x <= quintiles[c][.6]:
            return 3
        if x <= quintiles[c][.8]:
            return 4
        else:
            return 5

    rfm['R'] = rfm['Recency'].apply(lambda x : r_score(x))
    rfm['F'] = rfm['Frequency'].apply(lambda x : fm_score(x, 'Frequency'))
    rfm['M'] = rfm['Monetary'].apply(lambda x : fm_score(x, 'Monetary'))

    rfm['RFM_Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
    st.header('**RFM Score Table**')
    st.write(rfm)

    segt_map = {
        r'[1-2][1-2]': 'Lost',
        r'[1-2][3-4]': 'Sleeper',
        r'[1-2]5': 'Shouldn\'t Lose',
        r'3[1-2]': 'Cold Leads',
        r'33': 'need attention',
        r'[3-4][4-5]': 'loyal customers',
        r'41': 'Warm Leads',
        r'51': 'new customers',
        r'[4-5][2-3]': 'hopeful',
        r'5[4-5]': 'champions'
    }

    rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
    rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)
    ##-----Table------##
    st.header('**Segmentation Table**')
    st.write(rfm)

    # plot the distribution of customers over R and F
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    for i, p in enumerate(['R', 'F']):
        parameters = {'R':'Recency', 'F':'Frequency'}
        y = rfm[p].value_counts().sort_index()
        x = y.index
        ax = axes[i]
        bars = ax.bar(x, y, color='green')
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.set_title('Distribution of {}'.format(parameters[p]),
                    fontsize=14)
        for bar in bars:
            value = bar.get_height()
            if value == y.max():
                bar.set_color('blue')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value - 5,
                    '{}\n({}%)'.format(int(value), int(value * 100 / y.sum())),
                ha='center',
                va='top',
                color='w')

    #plt.show()
    plt.tight_layout()
    st.write(fig)

    # plot the distribution of M for RF score
    fig, axes = plt.subplots(nrows=5, ncols=5,
                            sharex=False, sharey=True,
                            figsize=(10, 10))

    r_range = range(1, 6)
    f_range = range(1, 6)
    for r in r_range:
        for f in f_range:
            y = rfm[(rfm['R'] == r) & (rfm['F'] == f)]['M'].value_counts().sort_index()
            x = y.index
            ax = axes[r - 1, f - 1]
            bars = ax.bar(x, y, color='green')
            if r == 5:
                if f == 3:
                    ax.set_xlabel('{}\nF'.format(f), va='top')
                else:
                    ax.set_xlabel('{}\n'.format(f), va='top')
            if f == 1:
                if r == 3:
                    ax.set_ylabel('R\n{}'.format(r))
                else:
                    ax.set_ylabel(r)
            ax.set_frame_on(False)
            ax.tick_params(left=False, labelleft=False, bottom=False)
            ax.set_xticks(x)
            ax.set_xticklabels(x, fontsize=8)

            for bar in bars:
                value = bar.get_height()
                if value == y.max():
                    bar.set_color('blue')
                ax.text(bar.get_x() + bar.get_width() / 2,
                        value,
                        int(value),
                        ha='center',
                        va='bottom',
                        color='k')
    fig.suptitle('Distribution of M for each F and R',
                fontsize=14)
    #plt.tight_layout()
    #st.write(fig)

    # count the number of customers in each segment
    segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots()

    bars = ax.barh(range(len(segments_counts)),
                segments_counts,
                color='green')
    ax.set_frame_on(False)
    ax.tick_params(left=False,
                bottom=False,
                labelbottom=False)
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)

    for i, bar in enumerate(bars):
            value = bar.get_width()
            if segments_counts.index[i] in ['champions', 'loyal customers']:
                bar.set_color('Blue')
            ax.text(value,
                    bar.get_y() + bar.get_height()/2,
                    '{:,} ({:}%)'.format(int(value),
                                    int(value*100/segments_counts.sum())),
                    va='center',
                    ha='left'
                )
    plt.tight_layout()
    st.write(fig)

    st.write('---') 
    st.header('**Apply K-means**')

    st.write('---')
    #st.header('**Profiling Report**')
    #st_profile_report(pr)
    #st.header('**Profiling Report**')
        #st_profile_report(pr)
    rfm = rfm.drop(["R","F","M","RFM_Score","Segment"], axis=1)
    st.header('**RFM Table**')
    st.write(rfm)
        #######
    fig, ax = plt.subplots()
    temp_rfm=rfm.drop(["CustomerID"], axis=1)
    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(1, 10))
    elbow.fit(temp_rfm)
    #elbow.show()
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(temp_rfm)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    #plt.show()  
    st.header('**The Elbow Method showing the optimal k**')
    plt.tight_layout()
    st.write(fig)

    #class_df = pd.read_excel('https://docs.google.com/spreadsheets/d/192dA4VZGReNwf-_PtuMYOG1RFWeHDwZY/edit?usp=share_link&ouid=111292030058453767559&rtpof=true&sd=true')
    #class_df.head()
    ##convert dataframe to numpy array

    #class_df = class_df.drop(["label"], axis=1)
    ##convert dataframe to numpy array
    #arr_rfm = class_df.to_numpy()
    arr_rfm = temp_rfm.to_numpy()
    ##st.write(type(arr_rfm))

    def k_means_elm(data, no_of_clusters):
        print("Running k-means")
        st.header('**ELM Model on K-Means**')
        data = np.array(data);
        kmeans = KMeans(no_of_clusters, random_state=0).fit_predict(data)
        #for i in range(0,178):
        #print(str(i) + " " + str(kmeans[i]));
        l1=kmeans[:90];
        l1.sort();
            
        l = [len(list(group)) for key, group in groupby(l1)]
        print (l);
        max1 = max(l);
        l1=kmeans[90:134];
        l1.sort();
            
        l = [len(list(group)) for key, group in groupby(l1)]
        print (l);
        max2 = max(l);
        l1=kmeans[134:178];
        l1.sort();
            
        l = [len(list(group)) for key, group in groupby(l1)]
        print (l);
        max3 = max(l);
        ##print("Clustering Accuracy = "+str(((max1+max2+max3)/181*100))+ " % ")
        st.write("Clustering Accuracy = "+str(((max1+max2+max3)/181*100))+ " % ")
        
    #ELM Model
    def regression_matrix(input_array,input_hidden_weights,bias):
        input_array = np.array(input_array);
        input_hidden_weights = np.array(input_hidden_weights);
        bias = np.array(bias);
        regression_matrix = np.add(np.dot(input_array,input_hidden_weights),bias);
        return regression_matrix;

    # Finding hidden layer activations
    def hidden_layer_matrix(regression_matrix):
        sigmoidal = [[0.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)];
        for i in range(0,no_of_inputs):
            for j in range(0,no_of_hidden_neurons):
                sigmoidal[i][j] = (1.0)/(1+math.exp(-(regression_matrix[i][j])))    
        return sigmoidal

    # Calculating the similarity matrix (S)
    def similarity_matrix():
        dist_array = [[0.0 for i in range(0,no_of_inputs)]for j in range(0,no_of_inputs)]
        for i in range(0,no_of_inputs):
            for j in range(0,no_of_inputs):
                for k in range(0,input_dim):
                    dist_array[i][j] +=  pow((input_array[i][k] - input_array[j][k]),2);
            
        for i in range(0, no_of_inputs):
            for j in range(0, no_of_inputs):
                dist_array[i][j] = math.exp((-(dist_array[i][j]))/(2*pow(sigma,2.0)));
        return dist_array;

    # Calculation of Graph Laplacian (L)
    def laplacian_matrix(similarity_matrix):
        diagonal_matrix = [[0.0 for i in range(0,no_of_inputs)]for j in range(0,no_of_inputs)];
        diagonal_matrix = np.array(diagonal_matrix);
        similarity_matrix = np.array(similarity_matrix);
        for i in range(0,no_of_inputs):
            for j in range(0,no_of_inputs):
                diagonal_matrix[i][i] += similarity_matrix[i][j];
            
        return np.subtract(diagonal_matrix,similarity_matrix);

    ## Test Accurency
    print("Running ELM")
    input_dim=3;
    # Loading Dataset
    data = arr_rfm

    # Min-Max Normalization 
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    input_array = min_max_scaler.fit_transform(data);


    #Parameter Input
    input_array = np.array(input_array);
    no_of_inputs = 181;
    no_of_input_neurons = input_dim;
    no_of_hidden_neurons = 200;
    no_of_output_neurons = 100;
    sigma = 1000
    input_hidden_weights = [[random.uniform(0,1) for i in 
                            range(0,no_of_hidden_neurons)]for j in range(0,no_of_input_neurons)]

    bias = [[1.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)]
    trade_off_parameter = 0.000000000000000000000000000001


    hidden_matrix  = np.array(hidden_layer_matrix(regression_matrix(input_array,input_hidden_weights,bias)))
    laplacian_matrix = np.array(laplacian_matrix(similarity_matrix()))
    intermediate = np.dot(np.dot(hidden_matrix.T,laplacian_matrix),hidden_matrix)

    a = [[0.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_hidden_neurons)];
    for i in range(0,no_of_hidden_neurons):
        for j in range(0,no_of_hidden_neurons):
            a[i][i] = 1.0;
    a = np.array(a);
    a = np.add(a,trade_off_parameter*intermediate);

    eig_value , eig_vector = np.linalg.eig(a);
    eig_vector = eig_vector.T;
    req_eigen_vectors = [[0.0 for i in range(0,no_of_hidden_neurons)] 
                        for j in range(0,no_of_output_neurons)];
    req_eigen_vectors = np.array(req_eigen_vectors);

    # Sorting the eigen vectors using the eigen values
    for i in range(0,len(eig_value)-1):
        for j in range(0,len(eig_value)-i-1):
            if(eig_value[j]>eig_value[j+1]):
                eig_value[j],eig_value[j+1]=eig_value[j+1],eig_value[j];
                eig_vector[j],eig_vector[j+1]=eig_vector[j+1],eig_vector[j];
                    
    # Finding n0 smallest eigen values
    for i in range(0,no_of_output_neurons):
        req_eigen_vectors[i] = eig_vector[i];
            
        req_eigen_vectors[i] = np.divide(req_eigen_vectors[i],np.linalg.norm(np.dot(hidden_matrix,req_eigen_vectors[i].T)))

    hidden_matrix = np.array(hidden_matrix);
    req_eigen_vectors = np.array(req_eigen_vectors);
    output_matrix = np.dot(hidden_matrix,(req_eigen_vectors.T));

    i=0;
    print("Final Weights")
    print(req_eigen_vectors)
    k_means_elm(output_matrix,elbow.elbow_value_);
        
    
else:
    st.info('Awaiting for excel file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        #@st.cache(allow_output_mutation=True)
        def load_data():
            #a = pd.DataFrame(
            #    np.random.rand(100, 5),
            #    columns=['a', 'b', 'c', 'd', 'e']
            #)
            url = "https://raw.githubusercontent.com/Rush3DLigth/Customer-Segmentation/main/dataset_by_part_v2.csv"
            s = requests.get(url).content
            a = pd.read_csv(io.StringIO(s.decode('utf-8')))
            a['InvoiceDate'] = a.InvoiceDate.astype('datetime64[s]') #change type object to datetime
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')


        st.header('**Group by Invoice No.**')
        df['Price'] = df['Quantity'] * df['UnitPrice']
        df_order = df.groupby(['InvoiceNo','InvoiceDate','CustomerID']).agg({'Price':lambda x:x.sum()}).reset_index()
        st.write(df_order.head())
        st.write(df_order.describe())
        NOW = df_order['InvoiceDate'].max() + timedelta(days=1)
        period = 365
        df_order['Recency'] = df_order['InvoiceDate'].apply(lambda x:(NOW - x).days)
        aggr = {'Recency':lambda x:x.min(), #the number of days since last order (Recency)
        'InvoiceDate':lambda x:len([d for d in x if d >= NOW - timedelta(days=period)]),} # the total number of order in the last period (Frequency)
        rfm = df_order.groupby('CustomerID').agg(aggr).reset_index()
        rfm.rename(columns={'InvoiceDate':'Frequency'},inplace=True)
        rfm['Monetary'] = rfm['CustomerID'].apply(lambda x:df_order[(df_order['CustomerID']==x) & (df_order['InvoiceDate'] >= NOW - timedelta(days=period))]['Price'].sum())
        st.header('**RFM Table**')
        st.write(rfm.head())
        st.header('**RFM Statistics**')
        st.write(rfm.describe())

        #Assige score to RFM
        quintiles = rfm[['Recency','Frequency','Monetary']].quantile([.2, .4, .6, .8]).to_dict() #ควอนไทล์แจกแจง
        def r_score(x):
            if x <= quintiles['Recency'][.2]:
                return 5
            elif x <= quintiles['Recency'][.4]:
                return 4
            elif x <= quintiles['Recency'][.6]:
                return 3
            elif x <= quintiles['Recency'][.8]:
                return 2
            else:
                return 1
        def fm_score(x, c):
            if x <= quintiles[c][.2]:
                return 1
            if x <= quintiles[c][.4]:
                return 2
            if x <= quintiles[c][.6]:
                return 3
            if x <= quintiles[c][.8]:
                return 4
            else:
                return 5

        rfm['R'] = rfm['Recency'].apply(lambda x : r_score(x))
        rfm['F'] = rfm['Frequency'].apply(lambda x : fm_score(x, 'Frequency'))
        rfm['M'] = rfm['Monetary'].apply(lambda x : fm_score(x, 'Monetary'))

        rfm['RFM_Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
        st.header('**RFM Score Table**')
        st.write(rfm)

        segt_map = {
            r'[1-2][1-2]': 'Lost',
            r'[1-2][3-4]': 'Sleeper',
            r'[1-2]5': 'Shouldn\'t Lose',
            r'3[1-2]': 'Cold Leads',
            r'33': 'need attention',
            r'[3-4][4-5]': 'loyal customers',
            r'41': 'Warm Leads',
            r'51': 'new customers',
            r'[4-5][2-3]': 'hopeful',
            r'5[4-5]': 'champions'
        }

        rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
        rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)
        ##-----Table------##
        st.header('**Segmentation Table**')
        st.write(rfm)

        # plot the distribution of customers over R and F
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        for i, p in enumerate(['R', 'F']):
            parameters = {'R':'Recency', 'F':'Frequency'}
            y = rfm[p].value_counts().sort_index()
            x = y.index
            ax = axes[i]
            bars = ax.bar(x, y, color='green')
            ax.set_frame_on(False)
            ax.tick_params(left=False, labelleft=False, bottom=False)
            ax.set_title('Distribution of {}'.format(parameters[p]),
                        fontsize=14)
            for bar in bars:
                value = bar.get_height()
                if value == y.max():
                    bar.set_color('blue')
                ax.text(bar.get_x() + bar.get_width() / 2,
                        value - 5,
                        '{}\n({}%)'.format(int(value), int(value * 100 / y.sum())),
                    ha='center',
                    va='top',
                    color='w')

        #plt.show()
        plt.tight_layout()
        st.write(fig)

        # plot the distribution of M for RF score
        fig, axes = plt.subplots(nrows=5, ncols=5,
                                sharex=False, sharey=True,
                                figsize=(10, 10))

        r_range = range(1, 6)
        f_range = range(1, 6)
        for r in r_range:
            for f in f_range:
                y = rfm[(rfm['R'] == r) & (rfm['F'] == f)]['M'].value_counts().sort_index()
                x = y.index
                ax = axes[r - 1, f - 1]
                bars = ax.bar(x, y, color='green')
                if r == 5:
                    if f == 3:
                        ax.set_xlabel('{}\nF'.format(f), va='top')
                    else:
                        ax.set_xlabel('{}\n'.format(f), va='top')
                if f == 1:
                    if r == 3:
                        ax.set_ylabel('R\n{}'.format(r))
                    else:
                        ax.set_ylabel(r)
                ax.set_frame_on(False)
                ax.tick_params(left=False, labelleft=False, bottom=False)
                ax.set_xticks(x)
                ax.set_xticklabels(x, fontsize=8)

                for bar in bars:
                    value = bar.get_height()
                    if value == y.max():
                        bar.set_color('blue')
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            value,
                            int(value),
                            ha='center',
                            va='bottom',
                            color='k')
        fig.suptitle('Distribution of M for each F and R',
                    fontsize=14)
        #plt.tight_layout()
        #st.write(fig)

        # count the number of customers in each segment
        segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

        fig, ax = plt.subplots()

        bars = ax.barh(range(len(segments_counts)),
                    segments_counts,
                    color='green')
        ax.set_frame_on(False)
        ax.tick_params(left=False,
                    bottom=False,
                    labelbottom=False)
        ax.set_yticks(range(len(segments_counts)))
        ax.set_yticklabels(segments_counts.index)

        for i, bar in enumerate(bars):
                value = bar.get_width()
                if segments_counts.index[i] in ['champions', 'loyal customers']:
                    bar.set_color('blue')
                ax.text(value,
                        bar.get_y() + bar.get_height()/2,
                        '{:,} ({:}%)'.format(int(value),
                                        int(value*100/segments_counts.sum())),
                        va='center',
                        ha='left'
                    )
        plt.tight_layout()
        st.write(fig)

        st.write('---') 
        st.header('**Apply ELM K-means**')
        st.write('---')
        #st.header('**Profiling Report**')
        #st_profile_report(pr)
        rfm = rfm.drop(["R","F","M","RFM_Score","Segment"], axis=1)
        st.header('**RFM Table**')
        st.write(rfm)
        #######
        fig, ax = plt.subplots()
        temp_rfm=rfm.drop(["CustomerID"], axis=1)
        kmeans = KMeans()
        elbow = KElbowVisualizer(kmeans, k=(1, 10))
        elbow.fit(temp_rfm)
        #elbow.show()
        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(temp_rfm)
            distortions.append(kmeanModel.inertia_)
        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        #plt.show()  
        st.header('**The Elbow Method showing the optimal k**')
        plt.tight_layout()
        st.write(fig)

        ##convert dataframe to numpy array
        #class_df = class_df.drop(["label"], axis=1)
        arr_rfm = temp_rfm.to_numpy()
        st.write(type(arr_rfm))

        def k_means_elm(data, no_of_clusters):
            print("Running k-means")
            st.header('**Running k-means**')
            data = np.array(data);
            kmeans = KMeans(no_of_clusters, random_state=0).fit_predict(data)
            #for i in range(0,178):
            #print(str(i) + " " + str(kmeans[i]));
            l1=kmeans[:90];
            l1.sort();
            
            l = [len(list(group)) for key, group in groupby(l1)]
            print (l);
            max1 = max(l);
            l1=kmeans[90:134];
            l1.sort();
            
            l = [len(list(group)) for key, group in groupby(l1)]
            print (l);
            max2 = max(l);
            l1=kmeans[134:178];
            l1.sort();
            
            l = [len(list(group)) for key, group in groupby(l1)]
            print (l);
            max3 = max(l);
            print("Clustering Accuracy = "+str(((max1+max2+max3)/178*100))+ " % ")
            st.write("Clustering Accuracy = "+str(((max1+max2+max3)/178*100))+ " % ")
        
        #ELM Model
        def regression_matrix(input_array,input_hidden_weights,bias):
            input_array = np.array(input_array);
            input_hidden_weights = np.array(input_hidden_weights);
            bias = np.array(bias);
            regression_matrix = np.add(np.dot(input_array,input_hidden_weights),bias);
            return regression_matrix;

        # Finding hidden layer activations
        def hidden_layer_matrix(regression_matrix):
            sigmoidal = [[0.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)];
            for i in range(0,no_of_inputs):
                for j in range(0,no_of_hidden_neurons):
                    sigmoidal[i][j] = (1.0)/(1+math.exp(-(regression_matrix[i][j])))    
            return sigmoidal

        # Calculating the similarity matrix (S)
        def similarity_matrix():
            dist_array = [[0.0 for i in range(0,no_of_inputs)]for j in range(0,no_of_inputs)]
            for i in range(0,no_of_inputs):
                for j in range(0,no_of_inputs):
                    for k in range(0,input_dim):
                        dist_array[i][j] +=  pow((input_array[i][k] - input_array[j][k]),2);
            
            for i in range(0, no_of_inputs):
                for j in range(0, no_of_inputs):
                    dist_array[i][j] = math.exp((-(dist_array[i][j]))/(2*pow(sigma,2.0)));
            return dist_array;

        # Calculation of Graph Laplacian (L)
        def laplacian_matrix(similarity_matrix):
            diagonal_matrix = [[0.0 for i in range(0,no_of_inputs)]for j in range(0,no_of_inputs)];
            diagonal_matrix = np.array(diagonal_matrix);
            similarity_matrix = np.array(similarity_matrix);
            for i in range(0,no_of_inputs):
                for j in range(0,no_of_inputs):
                    diagonal_matrix[i][i] += similarity_matrix[i][j];
            
            return np.subtract(diagonal_matrix,similarity_matrix);

        ## Test Accurency for Iris dataset
        print("Running ELM")
        input_dim=3;
        # Loading Dataset
        data = arr_rfm

        # Min-Max Normalization 
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        input_array = min_max_scaler.fit_transform(data);


        #Parameter Input
        input_array = np.array(input_array);
        no_of_inputs = 178;
        no_of_input_neurons = input_dim;
        no_of_hidden_neurons = 200;
        no_of_output_neurons = 100;
        sigma = 1000
        input_hidden_weights = [[random.uniform(0,1) for i in 
                                range(0,no_of_hidden_neurons)]for j in range(0,no_of_input_neurons)]

        bias = [[1.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)]
        trade_off_parameter = 0.000000000000000000000000000001


        hidden_matrix  = np.array(hidden_layer_matrix(regression_matrix(input_array,input_hidden_weights,bias)))
        laplacian_matrix = np.array(laplacian_matrix(similarity_matrix()))
        intermediate = np.dot(np.dot(hidden_matrix.T,laplacian_matrix),hidden_matrix)

        a = [[0.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_hidden_neurons)];
        for i in range(0,no_of_hidden_neurons):
            for j in range(0,no_of_hidden_neurons):
                a[i][i] = 1.0;
        a = np.array(a);
        a = np.add(a,trade_off_parameter*intermediate);

        eig_value , eig_vector = np.linalg.eig(a);
        eig_vector = eig_vector.T;
        req_eigen_vectors = [[0.0 for i in range(0,no_of_hidden_neurons)] 
                            for j in range(0,no_of_output_neurons)];
        req_eigen_vectors = np.array(req_eigen_vectors);

        # Sorting the eigen vectors using the eigen values
        for i in range(0,len(eig_value)-1):
            for j in range(0,len(eig_value)-i-1):
                if(eig_value[j]>eig_value[j+1]):
                    eig_value[j],eig_value[j+1]=eig_value[j+1],eig_value[j];
                    eig_vector[j],eig_vector[j+1]=eig_vector[j+1],eig_vector[j];
                    
        # Finding n0 smallest eigen values
        for i in range(0,no_of_output_neurons):
            req_eigen_vectors[i] = eig_vector[i];
            
            req_eigen_vectors[i] = np.divide(req_eigen_vectors[i],np.linalg.norm(np.dot(hidden_matrix,req_eigen_vectors[i].T)))

        hidden_matrix = np.array(hidden_matrix);
        req_eigen_vectors = np.array(req_eigen_vectors);

        output_matrix = np.dot(hidden_matrix,(req_eigen_vectors.T));

        i=0;
        print("Final Weights")
        print(req_eigen_vectors)

        k_means_elm(output_matrix,elbow.elbow_value_);
