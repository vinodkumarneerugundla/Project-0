#import all necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

#load the dataset
data=pd.read_csv('bikes.csv')
data.head()

#removing unwanted columns
data.drop(columns=['location'],inplace=True)
data.isnull().sum()

#removing the rows having null values
data.dropna(inplace=True)
data.duplicated().sum()
data.drop_duplicates(inplace=True)
data.duplicated().sum()

#checking unique columns
for col in data.columns:
    print(data[col].unique())

#function to convert multiple words to single words
def remove_words(value):
    value=value.split(' ')[0]
    return value.strip(' ')

#applying the function to the column
data['model_name']=data['model_name'].apply(remove_words)
data['model_name'].unique()

#converting categorical data to numerical data
data['model_name'].replace(['Bajaj', 'Royal', 'Hyosung', 'Jawa', 'KTM', 'TVS', 'Yamaha',
       'Honda', 'UM', 'Hero', 'Suzuki', 'Husqvarna', 'Mahindra',
       'Harley-Davidson', 'Kawasaki', 'Benelli', 'Triumph', 'Ducati',
       'BMW', '', 'BenelliImperiale', 'Moto', 'Fb', 'Indian', 'Yazdi',
       'Aprilia', 'MV'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],inplace=True)


data['model_name'].unique()
#converting categorical data to numerical data
data['kms_driven'].unique()
data['kms_driven'] = data['kms_driven'].str.strip().str.replace('\n\n', '', regex=False)


patterns_to_remove = r'(kmpl|Kmpl|km|Kms)$'

# Remove specified patterns
data['mileage'] = data['mileage'].str.replace(patterns_to_remove, '', regex=True).str.strip()

# Convert cleaned column to numeric (if applicable)
data['mileage'] = pd.to_numeric(data['mileage'], errors='coerce')

# Display the cleaned DataFrame
print(data)

data['mileage'].unique()

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

#removing extra content
kms_patterns_to_remove = r'(Km|Kms|Mileage\s\d+\sKms)$'

# Remove specified patterns
data['kms_driven'] = data['kms_driven'].str.replace(kms_patterns_to_remove, '', regex=True).str.strip()

# Convert cleaned column to numeric (if applicable)
data['kms_driven'] = pd.to_numeric(data['kms_driven'], errors='coerce')

# Display the cleaned DataFrame
print(data)
patterns_to_remove = r'(bhp|PS|kW|[^\d.])'

# Remove specified patterns
data['power'] = data['power'].str.replace(patterns_to_remove, '', regex=True).str.strip()

# Convert to numeric values
data['power'] = pd.to_numeric(data['power'], errors='coerce')

# Display the cleaned DataFrame
print(data)
data['owner']=data['owner'].apply(remove_words)

data['owner'].unique()

#removing extra content
data['owner'].replace(['first', 'third', 'second', 'fourth'],[1,2,3,4],inplace=True)
data.dropna(inplace=True)

#resetting the index
data.reset_index(inplace=True)

#dropping unwanted column index

data.drop(columns=['index'],inplace=True)

#taking input and output data from the dataset
input_data=data.drop(columns=['price'])
output_data=data['price']

#taking training and testing data
x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.2)

#training the model
model=LinearRegression()
model.fit(x_train,y_train)
predict=model.predict(x_test)
#passing input values to predict the bike price
input_data=pd.DataFrame([[3,2012,174676.0,3,45.0,34.0]],
                        columns=['model_name','model_year','kms_driven','owner','mileage','power'])
model.predict(input_data)

plt.figure(figsize=(30, 12))

# Scatter plot: Actual vs. Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predict, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs. Predicted Price')
plt.show()

# Scatter plot: kms_Driven vs. Price
plt.figure(figsize=(8, 6))
plt.scatter(data['kms_driven'], data['price'], color='green', alpha=0.5)
plt.xlabel('Kms Driven')
plt.ylabel('Price')
plt.title('Kms Driven vs. Price')
plt.show()

# Box plot for kms_driven
plt.subplot(1, 3, 1)
sns.boxplot(data['kms_driven'])
plt.title('Box Plot of Kms Driven')
plt.show()

#Box plot for mileage
plt.subplot(1, 3, 2)
sns.boxplot(data['mileage'])
plt.title('Box Plot of Mileage')
plt.show()

# Box plot for power
plt.subplot(1, 3, 3)
sns.boxplot(data['power'])
plt.title('Box Plot of Power')
plt.show()

#creating pickle file
import pickle as pkl
import joblib
joblib.dump(model, 'bike_price_model.pkl')


