#!/usr/bin/env python
# coding: utf-8

# In[170]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# loading data from csv file
crops_data = pd.read_csv('crops_prices_historical_datasets.csv')
crops_data


# In[171]:


# Select relevant features and target variable
relevant_features = crops_data[['Temperature', 'RainFall Annual', 'Regional_Demand', 'Crop', 'Price']]

# Check the first few rows
print(relevant_features)


# In[172]:


# Fill missing values (if any)
relevant_features.fillna(method='ffill', inplace=True)

# Encoding categorical variables
relevant_features = pd.get_dummies(relevant_features, columns=['Regional_Demand', 'Crop'], drop_first=True)

# Define features (X) and target variable (y)
X = relevant_features.drop('Price', axis=1)  # Features
y = relevant_features['Price']                 # Target variable

X


# In[173]:


y


# In[214]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
crops_data = pd.read_csv('crops_prices_historical_datasets.csv')

# Preprocess the data
relevant_features = crops_data[['Temperature', 'RainFall Annual', 'Regional_Demand', 'Crop', 'Price']]

# Fill missing values (if any)
relevant_features.fillna(method='ffill', inplace=True)

# Convert Price to categorical values
price_bins = [0, 20000, 40000, float('inf')]
price_labels = ['Low', 'Medium', 'High']
relevant_features['Price_Category'] = pd.cut(relevant_features['Price'], bins=price_bins, labels=price_labels)

# Encoding categorical variables
relevant_features = pd.get_dummies(relevant_features, columns=['Regional_Demand', 'Crop'], drop_first=True)

# Define features (X) and target variable (y)
X = relevant_features.drop(['Price', 'Price_Category'], axis=1)  # Features
y = relevant_features['Price_Category']                           # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(class_report)


# In[212]:


import joblib

# Save the model to a file
joblib_file = 'Crop_Price_Predictor_Model.h5'  # You can use .h5 but it's usually .pkl for joblib
joblib.dump(model, joblib_file)

print(f'Model saved as {joblib_file}')


# In[194]:


# import matplotlib.pyplot as plt
# import seaborn as sns

# # Group by 'Crop' and calculate the mean price for each crop
# average_price_per_crop = crops_data.groupby('Crop')['Price'].mean().sort_values()

# # Plot a bar chart to show the average price per crop
# plt.figure(figsize=(12, 6))
# sns.barplot(x=average_price_per_crop.index, y=average_price_per_crop.values, palette="viridis")

# # Set plot labels and title
# plt.title('Average Price of Each Crop', fontsize=16)
# plt.xlabel('Crop', fontsize=12)
# plt.ylabel('Average Price', fontsize=12)
# plt.xticks(rotation=90)  # Rotate crop names for better readability
# plt.tight_layout()

# # Show the plot
# plt.show()


# In[192]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define conversion rate from Rupees to Rands (approximate)
conversion_rate_inr_to_zar = 0.22

# Convert the 'Price' from Rupees to Rands
crops_data['Price_in_Rands'] = crops_data['Price'] * conversion_rate_inr_to_zar

# Plot a bar chart to show the actual price per crop in Rands
plt.figure(figsize=(12, 6))
sns.barplot(x=crops_data['Crop'], y=crops_data['Price_in_Rands'], palette="viridis")

# Set plot labels and title
plt.title('Crop Market Price Stats (in Rands)', fontsize=16)
plt.xlabel('Crop', fontsize=12)
plt.ylabel('Price (in Rands)', fontsize=12)
plt.xticks(rotation=90)  # Rotate crop names for better readability
plt.tight_layout()

# Show the plot
plt.show()


# In[200]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define conversion rate from Rupees to Rands (approximate)
conversion_rate_inr_to_zar = 0.22

# Convert the 'Price' from Rupees to Rands
crops_data['Price_in_Rands'] = crops_data['Price'] * conversion_rate_inr_to_zar

# Plot a line chart to show the actual price per crop in Rands
plt.figure(figsize=(12, 6))
sns.lineplot(x=crops_data['Crop'], y=crops_data['Price_in_Rands'], marker="o", color='b')

# Set plot labels and title
plt.title('Crop Market Price Stats (in Rands)', fontsize=16)
plt.xlabel('Crop', fontsize=12)
plt.ylabel('Price (in Rands)', fontsize=12)
plt.xticks(rotation=90)  # Rotate crop names for better readability
plt.tight_layout()

# Show the plot
plt.show()


# In[202]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define conversion rate from Rupees to Rands (approximate)
conversion_rate_inr_to_zar = 0.22

# Convert the 'Price' from Rupees to Rands
crops_data['Price_in_Rands'] = crops_data['Price'] * conversion_rate_inr_to_zar

# Sort the data by Crop for better plotting
crops_data_sorted = crops_data.sort_values('Crop')

# Plot an area chart to show the actual price per crop in Rands
plt.figure(figsize=(12, 6))
plt.fill_between(crops_data_sorted['Crop'], crops_data_sorted['Price_in_Rands'], color="skyblue", alpha=0.4)
plt.plot(crops_data_sorted['Crop'], crops_data_sorted['Price_in_Rands'], color="Slateblue", alpha=0.6)

# Set plot labels and title
plt.title('Crop Market Price Stats (in Rands)', fontsize=16)
plt.xlabel('Crop', fontsize=12)
plt.ylabel('Price (in Rands)', fontsize=12)
plt.xticks(rotation=90)  # Rotate crop names for better readability
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:




