import numpy as np
import pandas as pd
from  textblob import TextBlob
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("C:\\Users\\aryan\\Downloads\\37000_reviews_of_thread_app.csv\\37000_reviews_of_thread_app.csv")
df

rows, cols = df.shape
print(f'There are {rows} rows and {cols} columns in dataset')
print(f"There are {df.duplicated().sum()} duplicate values")

df.isna().sum()

df.drop(['review_title','developer_response','developer_response_date','review_id','user_name','appVersion','appVersion','laguage_code','country_code','review_date','Unnamed: 0'],axis=1,inplace = True)#axis-->vertical axis, inplace-->making changes in original dataframe
df.reset_index(drop=True,inplace=True) #delete rows or colums from original df
df.head()
df.info()
df.describe().T
df.describe(include = 'object').T

def analyze(x):
    if x>=0.5:
        return 'Positive'
    elif x<=-0.5:
        return 'Negative'
    else:
        return 'Neutral'

def score(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity
df['score'] = df['review_description'].apply(score)
df['analysis'] = df['score'].apply(analyze)
df.head()

def rating(x):

    if x>=4:
        return 'Positive'
    elif x<=2:
        return 'Negative'
    else:
        return 'Neutral'

df['rating_analysis'] = df['rating'].apply(rating)
df.head()
df['final_rating'] = 'Positive'

for i in range(len(df)):
    if (df['analysis'][i]=='Positive' and df['rating_analysis'][i] == 'Positive'):
        df['final_rating'][i] = 'Positive'

    elif (df['analysis'][i] == 'Positive' and df['rating_analysis'][i] == 'Neutral')or (df['analysis'][i] == 'Neutral' and df['rating_analysis'][i] == 'Positive'):
        df['final_rating'][i] = 'Positive'

    elif (df['analysis'][i] == 'Negative' and df['rating_analysis'][i] == 'Neutral')or (df['analysis'][i] == 'Neutral' and df['rating_analysis'][i] == 'Negative'):
        df['final_rating'][i] = 'Negative'

    elif (df['analysis'][i] == 'Neutral' and df['rating_analysis'][i] == 'Neutral'):
        df['final_rating'][i] = 'Neutral'

    elif (df['analysis'][i] == 'Negative' and df['rating_analysis'][i] == 'Negative'):
        df['final_rating'][i] = 'Negative'
    else:
        df['final_rating'][i] = 'Neutral'
df.head()
df['final_rating'].unique()
df.describe(include='object')
sns.countplot(x='final_rating',data=df)