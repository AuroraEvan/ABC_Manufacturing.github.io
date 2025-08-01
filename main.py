# Streamlit App: Data-Driven Solutions for ABC Manufacturing (P7)

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Enhanced CSS for a professional and user-friendly interface
st.markdown(
    """
    <style>
    .main {
        background-color: #f9fafb;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stApp {
        background-color: #ffffff;
    }
    .title {
        color: #164e63;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        background-color: #cffafe;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header {
        color: #1e293b;
        font-size: 24px;
        font-weight: bold;
        margin-top: 25px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 5px;
    }
    .text {
        color: #374151;
        font-size: 16px;
        line-height: 1.8;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #164e63;
        color: white;
        border-radius: 8px;
        padding: 12px 25px;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e3a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stDataFrame {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        background-color: #f9fafb;
    }
    .chart-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown('<div class="title">Data-Driven Solutions for ABC Manufacturing (P7)</div>', unsafe_allow_html=True)

# Introduction and LO3 (P5): Tools and Technologies
st.markdown('<div class="text">'
            'As a Junior Analyst at ABC Manufacturing, this application leverages data science to enhance supply chain '
            'management and operational efficiency. It utilizes tools such as Pandas for data processing, Matplotlib and '
            'Seaborn for visualizations, Scikit-learn for machine learning models (KMeans and Linear Regression), and '
            'Streamlit for an interactive interface. These technologies support business processes by enabling demand '
            'forecasting, real-time equipment monitoring via IoT, quality control through data analysis, and seamless '
            'collaboration with suppliers, aligning with the company’s goals to optimize resources and improve decision-making.'
            '</div>', unsafe_allow_html=True)

# LO3 (M3): Benefits of Data Science
st.markdown('<div class="text">'
            'Implementing data science provides ABC Manufacturing with tangible benefits, including precise demand '
            'forecasting to reduce overstocking, real-time insights from IoT devices to minimize production downtime, '
            'and quality analytics to prevent costly recalls. These advantages enhance operational efficiency, lower '
            'costs, and boost customer satisfaction, addressing critical real-world challenges in the supply chain.'
            '</div>', unsafe_allow_html=True)

# Step 1: Data Collection
st.markdown('<div class="header">Step 1: Data Collection</div>', unsafe_allow_html=True)
st.markdown('<div class="text">This application automatically analyzes the **marketing_campaign.csv** dataset, '
            'sourced from ABC Manufacturing’s customer data, to support strategic decision-making.</div>', unsafe_allow_html=True)

# Read data from GitHub or local file
data_url = "https://raw.githubusercontent.com/username/p7-customer-analysis/main/marketing_campaign.csv"
try:
    data = pd.read_csv(data_url, sep='\t')
except Exception as e:
    st.error(f"Failed to load data from URL: {str(e)}. Using local file as fallback.")
    data = pd.read_csv("marketing_campaign.csv", sep='\t')

st.markdown('<div class="text">**Data Dimensions:**</div>', unsafe_allow_html=True)
st.write(data.shape)
st.markdown('<div class="text">**Column Information:**</div>', unsafe_allow_html=True)
st.write(data.info())
st.markdown('<div class="text">**Data Sample:**</div>', unsafe_allow_html=True)
st.dataframe(data.head())

# Step 2: Data Preprocessing
st.markdown('<div class="header">Step 2: Data Preprocessing</div>', unsafe_allow_html=True)

data['Income'].fillna(data['Income'].mean(), inplace=True)
data = data.drop_duplicates()
st.markdown('<div class="text">Number of records after removing nulls and duplicates:</div>', unsafe_allow_html=True)
st.write(len(data))

data = data[data['Income'] < 200000]
st.markdown('<div class="text">Number of records after removing outliers:</div>', unsafe_allow_html=True)
st.write(len(data))
st.markdown('<div class="text">Average Income:</div>', unsafe_allow_html=True)
st.write(data['Income'].mean())

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
data['Years_Since_Customer'] = (pd.Timestamp('2025-07-30') - data['Dt_Customer']).dt.days / 365.25
data['Total_Spend'] = (data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] +
                       data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'])
data['Campaign_Acceptance'] = data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                                    'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis=1)
st.markdown('<div class="text">Years since customer registration (min-max):</div>', unsafe_allow_html=True)
st.write(data['Years_Since_Customer'].min(), "-", data['Years_Since_Customer'].max())
st.markdown('<div class="text">Average Total Spend:</div>', unsafe_allow_html=True)
st.write(data['Total_Spend'].mean())
st.markdown('<div class="text">Average Campaign Acceptance:</div>', unsafe_allow_html=True)
st.write(data['Campaign_Acceptance'].mean())

features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
            'NumStorePurchases', 'NumWebVisitsMonth', 'Years_Since_Customer']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
st.markdown('<div class="text">Standardized data (sample):</div>', unsafe_allow_html=True)
st.write(data_scaled[:2])

# Step 3: Data Analysis
st.markdown('<div class="header">Step 3: Data Analysis</div>', unsafe_allow_html=True)

st.markdown('<div class="text">**Income Statistics:**</div>', unsafe_allow_html=True)
st.write(data['Income'].describe())
st.markdown('<div class="text">**Total Spend Statistics:**</div>', unsafe_allow_html=True)
st.write(data['Total_Spend'].describe())
st.markdown('<div class="text">**Campaign Acceptance Rate:**</div>', unsafe_allow_html=True)
st.write(data['Campaign_Acceptance'].value_counts(normalize=True))

# Step 4: Data Visualization
st.markdown('<div class="header">Step 4: Data Visualization</div>', unsafe_allow_html=True)

kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=data['Income'], y=data['Total_Spend'], hue=data['Cluster'], 
                size=data['Campaign_Acceptance'], sizes=(20, 200), palette='viridis', ax=ax)
plt.title('Customer Segmentation: Income vs Total Spend')
plt.xlabel('Income (USD)')
plt.ylabel('Total Spend (USD)')
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='Cluster', y='Income', data=data, ax=ax)
plt.title('Income Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Income (USD)')
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                 'MntSweetProducts', 'MntGoldProds']
spending_by_cluster = data.groupby('Cluster')[spending_cols].mean()
fig, ax = plt.subplots(figsize=(12, 6))
spending_by_cluster.plot(kind='bar', stacked=True, ax=ax)
plt.title('Average Spending by Product and Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Spending (USD)')
plt.legend(title='Product')
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

corr_cols = ['Income', 'Recency', 'Total_Spend', 'NumWebPurchases', 'NumStorePurchases']
corr_matrix = data[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
plt.title('Correlation Heatmap')
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6))
for cluster in range(4):
    cluster_data = data[data['Cluster'] == cluster]
    ax.plot(cluster_data
