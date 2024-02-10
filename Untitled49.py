#!/usr/bin/env python
# coding: utf-8

# In[28]:


from itertools import count
from nltk.util import pr
import pandas as pd
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/user_reviews.csv"
data = pd.read_csv(url)
print(data.head())


# In[29]:


print(data.isnull().sum())


# In[30]:


data = data.dropna()
print(data.isnull().sum())


# In[32]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
data["positive"] = [sentiments.polarity_scores(i)["pos"] for  i in data["Translated_Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"]  for  i in data["Translated_Review"]]
data["Neutral"] =  [sentiments.polarity_scores(i)["neu"]  for  i in data["Translated_Review"]]
print(data.head())


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15, 10))
sns.scatterplot(data['Sentiment_Polarity'],data['Sentiment_Subjectivity']
           ,hue = data['Sentiment']     )
plt.title("Google Play Store Reviews Sentiment Analysis", fontsize=20)
plt.show()


# In[ ]:




