#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # To avoid the warnings

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Load the dataset

# In[3]:


df=pd.read_csv("youtubers_df.csv")
df


# # 1. Data Exploration:
# 
# Load the dataset into a Pandas DataFrame.
# Use methods like head(), info(), describe() to get an initial understanding of the dataset.
# Check for missing values and outliers using methods like isnull(), sum(), and visualization tools.

# In[4]:


# Check the structure of the dataset
print(df.info())


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


# Display the shape of the dataset
df.shape


# In[8]:


# Display the size of the dataset
df.size


# In[9]:


# Statistical Summary of all columns
df.iloc[:, :-1].describe().T.sort_values(by='std', ascending = False)\
                          .style.background_gradient(cmap="Greens")\
                          .bar(subset=["max"], color='#F8766D')\
                          .bar(subset=["mean"], color='#00BFC4')


# In[10]:


# Check for missing data
df.isnull().sum()


# In[11]:


#Checking the missing values
df.isnull().sum().sort_values()


# In[12]:


# Commonly Replacing the missing values with one varible
for Col in df.columns:
    if df[Col].dtype == 'object':
        df[Col].fillna('Entertainment' , inplace = True) 
    else:
        df[Col].fillna(df[Col].mean(), inplace = True)


# In[13]:


#Checking the missing values
df.isnull().sum().sort_values()


# In[14]:


# Check for outliers
plt.figure(figsize = (12,8))
sns.boxplot(df)
plt.grid()
plt.show()


# In[15]:


# Remove outliers from the 'Subscribers' column
Q1 = df['Suscribers'].quantile(0.25)
Q3 = df['Suscribers'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Suscribers'] > lower_bound) & (df['Suscribers'] < upper_bound)]
# Remove outliers from the 'Visits' column
Q1 = df['Visits'].quantile(0.25)
Q3 = df['Visits'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Visits'] > lower_bound) & (df['Visits'] < upper_bound)]


# In[16]:


# Check for outliers after removal
plt.figure(figsize = (12,8))
sns.boxplot(df)
plt.grid()
plt.show()


# # 2. Trend Analysis:
# 
# Group data by categories and analyze trends using tools like line plots or bar charts.
# Use correlation analysis to investigate relationships between variables.

# In[17]:


# Identify trends among the top YouTube streamers
top_categories = df.groupby('Categories')['Suscribers'].sum().nlargest(5)
print(top_categories)


# In[18]:


# Correlation analysis
correlation_matrix = df[['Suscribers', 'Likes', 'Comments']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# # 3. Audience Study:
# 
# Analyze the distribution of audiences by country using bar charts or maps.
# Identify regional preferences for specific content categories.

# In[19]:


# Analyze the distribution of streamers' audiences by country
audience_distribution = df['Country'].value_counts()
print(audience_distribution)


# In[20]:


# Analyze the distribution of streamers' audiences by country
audience_distribution = df['Country'].value_counts()
# Bar chart for audience distribution by country
plt.figure(figsize=(12, 6))
audience_distribution.plot(kind='bar', color='skyblue')
plt.title('Audience Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Number of Streamers')
plt.show()


# In[21]:


# 4. Identify regional preferences for specific content categories
regional_preferences = df.groupby(['Country', 'Categories']).size().unstack()

# Heatmap for regional preferences
plt.figure(figsize=(14, 8))
sns.heatmap(regional_preferences, cmap='YlGnBu', annot=True, fmt='g', cbar_kws={'label': 'Number of Streamers'})
plt.title('Regional Preferences for Content Categories')
plt.xlabel('Content Category')
plt.ylabel('Country')
plt.show()


# # 4. Performance Metrics:
# 
# Calculate and visualize average subscribers, visits, likes, and comments.
# Look for patterns or anomalies in the metrics using time series analysis or other appropriate methods.

# In[22]:


# Calculate average metrics
average_metrics = df[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

# Bar chart for average metrics
plt.figure(figsize=(10, 6))
average_metrics.plot(kind='bar', color=['blue', 'orange', 'green', 'red'])
plt.title('Average Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.show()


# In[23]:


# Assuming there is no specific date column, you can use index as a proxy for time
df.set_index(df.index, inplace=True)

# Line plot for trends over time
plt.figure(figsize=(12, 6))
df[['Suscribers', 'Visits', 'Likes', 'Comments']].plot()
plt.title('Performance Metrics Over Time')
plt.xlabel('Index (Proxy for Time)')
plt.ylabel('Count')
plt.show()


# # 5. Content Categories:
# 
# Explore the distribution of content categories using bar charts or pie charts.
# Identify categories with the highest number of streamers and exceptional performance metrics.

# In[24]:


# Explore the distribution of content categories using bar charts or pie charts
category_distribution = df['Categories'].value_counts()
# Bar chart for content category distribution
plt.figure(figsize=(12, 6))
category_distribution.plot(kind='bar', color='skyblue')
plt.title('Content Category Distribution')
plt.xlabel('Content Category')
plt.ylabel('Number of Streamers')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# Pie chart for content category distribution
plt.figure(figsize=(10, 10))
plt.pie(category_distribution, labels=category_distribution.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Content Category Distribution')
plt.show()


# In[26]:


# Identify categories with the highest number of streamers
top_categories_streamers = category_distribution.nlargest(5)
print(f'Top Categories by Number of Streamers:\n{top_categories_streamers}')


# In[27]:


# Identify categories with the highest number of streamers
top_categories_streamers = category_distribution.nlargest(5)
print(f'Top Categories by Number of Streamers:\n{top_categories_streamers}')


# In[28]:


# Identify categories with exceptional performance metrics (using mean of numeric columns)
numeric_columns = ['Suscribers', 'Visits', 'Likes', 'Comments']
exceptional_performance_categories = df.groupby('Categories')[numeric_columns].mean().nlargest(5, 'Suscribers')
print(f'Categories with Exceptional Subscribers:\n{exceptional_performance_categories}')


# # 6. Brands and Collaborations:
# 
# Analyze whether high-performing streamers receive more brand collaborations.
# Use visualizations to illustrate the relationship between performance metrics and collaborations.

# In[29]:


# Scatter plot for Subscribers vs Likes
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Suscribers', y='Likes', data=df, color='lightgreen')
plt.title('Suscribers vs Likes')
plt.xlabel('Number of Subscribers')
plt.ylabel('Number of Likes')
plt.show()


# In[30]:


# Scatter plot for Subscribers vs Comments
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Suscribers', y='Comments', data=df, color='orange')
plt.title('Subscribers vs Comments')
plt.xlabel('Number of Subscribers')
plt.ylabel('Number of Comments')
plt.show()


# # 7. Benchmarking:
# 
# Identify streamers with above-average performance.
# Create benchmarks based on various performance metrics.
# Visualize and compare streamers against benchmarks.

# In[31]:


# Identify streamers with above-average performance
average_metrics = df[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

# Create benchmarks based on various performance metrics
benchmark_subscribers = 1.5 * average_metrics['Suscribers']
benchmark_visits = 1.5 * average_metrics['Visits']
benchmark_likes = 1.5 * average_metrics['Likes']
benchmark_comments = 1.5 * average_metrics['Comments']


# In[32]:


print("benchmark_subscribers :",benchmark_subscribers)
print("benchmark_visits :",benchmark_visits)
print("benchmark_likes :",benchmark_likes)
print("benchmark_comments :", benchmark_comments)


# In[33]:


# Identify streamers with above-average performance
above_average_streamers = df[
    (df['Suscribers'] > benchmark_subscribers) &
    (df['Visits'] > benchmark_visits) &
    (df['Likes'] > benchmark_likes) &
    (df['Comments'] > benchmark_comments)
]


# In[34]:


# Bar chart to visualize and compare streamers against benchmarks
plt.figure(figsize=(12, 6))
metrics_columns = ['Suscribers', 'Visits', 'Likes', 'Comments']
for metric in metrics_columns:
    plt.bar(metric, average_metrics[metric], label='Average', color='skyblue', alpha=0.5)
    plt.bar(metric, above_average_streamers[metric].mean(), label='Above Average', color='lightgreen')

plt.title('Streamers vs Benchmarks')
plt.xlabel('Performance Metrics')
plt.ylabel('Count')
plt.legend()
plt.show()


# # 8. Content Recommendations:
# 
# Propose a system for enhancing content recommendations. This may involve collaborative filtering, content-based filtering, or hybrid methods.
# Consider factors like user preferences, historical data, and streamer characteristics.

# In[35]:


# Display visualizations
plt.figure(figsize=(12, 6))

# Example: Bar chart for content category distribution
plt.subplot(2, 3, 1)
category_distribution.plot(kind='bar', color='#FFC0CB')
plt.title('Content Category Distribution')
plt.xlabel('Content Category')
plt.ylabel('Number of Streamers')

# Example: Scatter plot for Subscribers vs Likes
plt.subplot(2, 3, 2)
sns.scatterplot(x='Suscribers', y='Likes', data=df, color='lightgreen')
plt.title('Subscribers vs Likes')
plt.xlabel('Number of Subscribers')
plt.ylabel('Number of Likes')

# Add more visualizations as needed

plt.tight_layout()
plt.show()

Enhancing content recommendations involves creating a personalized system that takes into account user preferences, historical data, and streamer characteristics. Here's a proposal that combines collaborative filtering and content-based filtering, forming a hybrid recommendation system:

Hybrid Recommendation System Proposal:
1. Collaborative Filtering:
User-Based Collaborative Filtering:

Identify users with similar content consumption patterns.
Recommend content that similar users have enjoyed but the current user hasn't seen.
Item-Based Collaborative Filtering:

Identify content (streamers or videos) with similar engagement patterns.
Recommend content that is similar to what the user has enjoyed.
2. Content-Based Filtering:
Utilize information about the content and the user's preferences:

Extract features such as streamer category, content type, historical user engagement, streamer popularity, etc.
Build a content profile and a user profile based on these features.
Recommend content that aligns with the user's historical preferences and content characteristics.

3. Hybrid Approach:
Combine collaborative and content-based filtering to provide more accurate and diverse recommendations.
Use weighted averages or ensemble methods to blend predictions from both approaches.
4. User and Streamer Characteristics:
Consider additional user and streamer characteristics:
User demographics, preferences, and historical interactions.
Streamer statistics such as subscriber count, content category, frequency of content updates, and collaboration history.
5. Implicit Feedback and Engagement Metrics:
Incorporate implicit feedback, such as the time spent on a video, clicks, and watch history, to better understand user preferences.
6. Dynamic Personalization:
Implement a dynamic recommendation system that adapts to changing user preferences over time.
7. Evaluation and Optimization:
Regularly evaluate and optimize the recommendation system using metrics like precision, recall, and user satisfaction.
Collect user feedback to continuously improve the system.
8. Experimentation:
Implement A/B testing to evaluate the impact of changes to the recommendation algorithm.
Experiment with different algorithms, feature sets, and weighting strategies.
9. Scalability and Real-Time Processing:
Design the system to be scalable, supporting a growing user and content base.
Consider real-time processing to provide up-to-date recommendations.
10. Privacy Considerations:
Implement privacy-aware algorithms to ensure user data is handled responsibly.
Conclusion:
This hybrid recommendation system considers collaborative filtering, content-based filtering, and additional factors to provide personalized and diverse content recommendations. It aims to strike a balance between capturing user preferences and leveraging content characteristics and streamer information for more accurate suggestions. Regular evaluation, optimization, and experimentation are crucial for maintaining and improving the effectiveness of the recommendation system.
# In[ ]:




