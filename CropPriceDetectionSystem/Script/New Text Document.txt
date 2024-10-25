# predict.py
import sys
import pandas as pd
from keras.models import load_model
import numpy as np
import json

# Load the model
model = load_model('Crop_Price_Predictor_Model.h5')

# Read input data from command line arguments
input_data = json.loads(sys.argv[1])  # Expecting a JSON string
input_df = pd.DataFrame([input_data])

# Make predictions
predictions = model.predict(input_df)
predicted_price_category = np.argmax(predictions, axis=1)  # Modify this based on your output

# Print the predictions
print(predicted_price_category[0])  # Output the predicted category
