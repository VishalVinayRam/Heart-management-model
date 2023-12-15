import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
fit_path = "fit_file_csv"
fit_test_path = "fit_file_test_csv"
graph_path = "graphs"

df = pd.read_csv('fit_file_csv/RUN_2021-03-25-10-40-55.fit.csv')
df['alt_difference'] = df['enhanced_altitude'] - df['enhanced_altitude'].shift(1)
df['rolling_ave_alt'] = df['alt_difference'].rolling(window=5).mean()
df = df.bfill()
df = df.drop(['position_lat','position_long'], axis=1, errors='ignore')
df.to_csv('fit_file_csv/RUN_2021-03-25-10-40-55.fit.csv')

    
# Assuming 'heart_rate' is the target variable to be predicted
target_variable = 'heart_rate'

# Select relevant features for the model
features = ['timestamp', 'lap', 'distance', 'enhanced_speed', 'power']

# Create a new DataFrame with selected features
df = df[features + [target_variable]]

# Convert timestamp to datetime and set it as the index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Define a function to create sequences for LSTM
def create_sequences(data, target_variable, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length].values
        target = data[target_variable].iloc[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Choose the sequence length (number of time steps to look back)
sequence_length = 10

# Create sequences and targets
X, y = create_sequences(df_scaled, target_variable, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer with 1 neuron for regression
model.compile(optimizer='adam', loss='mse')  # Mean Squared Error for regression task
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Save the model for later use
model.save('lstm_model.h5')