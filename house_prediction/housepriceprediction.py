import pandas as pd

df= pd.read_csv('bengaluru_house_prices.csv')

df1=df.drop(['area_type','society','balcony','availability'], axis=1)

import math as mt
df1['bath']= df1['bath'].fillna(mt.floor((df1['bath'].mean())))


df2= df1.dropna()


df2['size']=df2['size'].apply(lambda x: int(x.split(' ')[0]))

def isfloat(x):
    if x is None:
        return False
    try:
        float(x)
    except:
        return False
    return True




def convert_to_sqft(x):
    # Unit conversion constants
    CONVERSIONS = {
        'Sq. Meter': 10.764,  # 1 sq meter = 10.764 sq ft
        'Perch': 272.25,      # 1 perch = 272.25 sq ft
        'Cents': 435.6,       # 1 cent = 435.6 sq ft
        'Acres': 43560,       # 1 acre = 43560 sq ft
        'Grounds': 2400,      # 1 ground = 2400 sq ft
        'Guntha': 1089 ,       # 1 guntha = 1089 sq ft
        'Sq. Yards': 9          # 1 sq yard = 9 sq ft
    }
    
    # Handle None/NaN values
    if pd.isna(x):
        return None
        
    # Convert to string for processing
    x = str(x).strip()
    
    # Handle range values (e.g., "1500-2000")
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
        
    # Handle unit conversions
    for unit, multiplier in CONVERSIONS.items():
        if unit in x:
            try:
                value = float(x.replace(unit, '').strip())
                return value * multiplier
            except ValueError:
                ###print(f"Conversion error for value: {x}")
                continue
    # Try direct float conversion
    try:
        return float(x)
    except ValueError:
        return None

# Apply the conversion
df3 = df2.copy()

df3['total_sqft'] = df3['total_sqft'].apply(convert_to_sqft)


df4=df3.copy()

df4['price_per_sqft'] = df4['price']*100000 / df4['total_sqft']


location_stats=df4.groupby('location')['location'].agg('count').sort_values(ascending=False)


location_stats_less_than_10 = location_stats[location_stats < 10]
df4['location']=df4['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)


df4['location']=df4['location'].str.strip()

df5=df4[~(df4['total_sqft']/df4['size']<300)]


import numpy as np

def remove_outliers_sqft(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean=np.mean(subdf['price_per_sqft'])
        srd=np.std(subdf['price_per_sqft'])
        reduced_df=subdf[(subdf.price_per_sqft> (mean-srd)) & (subdf.price_per_sqft<(mean+srd))]
        df_out=pd.concat([df_out,reduced_df], ignore_index=True)
    return df_out
df6=remove_outliers_sqft(df5)



import matplotlib
import matplotlib.pyplot as plt
def plot_Scatter(df,location):
    bhk2=df[(df['location']==location) & (df['size']==2)]
    bhk3=df[(df['location']==location) & (df['size']==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price_per_sqft,color='blue',label='2 BHK')
    plt.scatter(bhk3.total_sqft,bhk3.price_per_sqft,color='red',label='3 BHK')
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(f"Price Per Square Feet vs Total Square Feet Area for {location}")
    plt.legend()
    plt.show()


def remove_bhk_outlier(df):
    exclude_indices= np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('size'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('size'):
            stats=bhk_stats.get(bhk-1)
            if(stats and stats['count']>5):
                exclude_indices=np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis=0)

df7=remove_bhk_outlier(df6)

df8=df7[(df7['bath']<=df7['size']+2)]



df9=df8.drop(['price_per_sqft'], axis=1)
print("Shape of df9 before one-hot encoding:", df9.shape)
print(df9.head())

df10=pd.get_dummies(df9,columns=['location'], drop_first=True,dtype='int')
df11=df10.drop(['location_other'], axis=1)

print(df11.head())




print("SQ_FT\n",df11['total_sqft'].describe())
print("Bath\n",df11['bath'].describe())
print("size\n",df11['size'].describe())
print("shape of d11 before removing outliers\n",df11.shape)
from scipy import stats
z_score_prices=stats.zscore(df11['price'])
df11['z_score_price'] = z_score_prices

z_score_total_sqft=stats.zscore(df11['total_sqft'])
df11['z_score_total_sqft'] = z_score_total_sqft

z_score_bath=stats.zscore(df11['bath'])
df11['z_score_bath'] = z_score_bath

z_score_size=stats.zscore(df11['size'])
df11['z_score_size'] = z_score_size


df12 = df11[
    (df11['z_score_price'].abs() < 2) & 
    (df11['z_score_total_sqft'].abs() < 2) & 
    (df11['z_score_bath'].abs() < 3) & 
    (df11['z_score_size'].abs() < 3)
]
print("Shape of df12 after removing outliers based on z-score:", df12.shape)
print("Price\n",df12['price'].describe())
print("SQ_FT\n",df12['total_sqft'].describe())
print("Bath\n",df12['bath'].describe())
print("size\n",df12['size'].describe())
df12= df12.drop(['z_score_price', 'z_score_total_sqft', 'z_score_bath', 'z_score_size'], axis=1)
x=df12.drop(['price'], axis=1)
y=df12['price']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



lr.fit(x_train, y_train)
score=lr.score(x_test, y_test)
print(f"Test score of Linear Regression: {score}")

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores=cross_val_score(lr,x,y,cv=ss)
print(f"Cross-validation score of Linear Regression: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")
import joblib
joblib.dump(lr,'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'")

import json
columns={
    'data_columns': [col for col in x.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))
print("Column names saved as 'columns.json'")

def get_EstimatedPrice(location, size, total_sqft, bath):
    loc='location_'+location.strip()
    try:
        loc_index=x.columns.get_loc(loc)
    except:
        loc_index = -1
    print("loc_index:", loc_index)
    a = np.zeros(len(x.columns))
    a[0] = size
    a[1] = total_sqft
    a[2] = bath
    if loc_index >=0:
        a[loc_index]=1
    print(loc_index)
    print("a[loc_index]:", a[loc_index])
    print("x.columns[loc_index]:", x.columns[loc_index])
    return round(lr.predict([a])[0],2)
print(get_EstimatedPrice('1st Phase JP Nagar', 5, 1500, 2))