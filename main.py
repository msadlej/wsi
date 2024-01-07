import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
)
from neural_network import NeuralNetwork, NeuronLayer

# Initialize an empty DataFrame
all_data = pd.DataFrame()

# Loop over the file names
for i in range(1, 71):
    # Generate the file name
    file_name = f"Diabetes-Data/data-{i:02d}"

    # Read the file into a DataFrame
    df = pd.read_csv(file_name, delimiter="\t", header=None)

    # Assign column names
    df.columns = ["Date", "Time", "Code", "Value"]

    # Append the data to the main DataFrame
    all_data = pd.concat([all_data, df], ignore_index=True)

# Drop the Date column
all_data = all_data.drop(columns=["Date"])

# Convert Time to minutes past midnight
all_data["Time"] = all_data["Time"].apply(
    lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]) if ":" in x else 0
)

# Get the unique 'Code' values and sort them
unique_codes = sorted(all_data["Code"].unique())

# Create a dictionary that maps each unique 'Code' to an integer from 0 to 19
code_map = {code: i for i, code in enumerate(unique_codes)}

# Replace the 'Code' values in the DataFrame using the dictionary
all_data["Code"] = all_data["Code"].map(code_map)

# Normalize the 'Time' and 'Value' columns to the range [0, 1]
all_data["Time"] = (all_data["Time"] - all_data["Time"].min()) / (
    all_data["Time"].max() - all_data["Time"].min()
)
all_data["Value"] = (all_data["Value"] - all_data["Value"].min()) / (
    all_data["Value"].max() - all_data["Value"].min()
)

# Apply sigmoid function to the 'Time' and 'Value' columns
all_data["Time"] = all_data["Time"].apply(lambda x: 1 / (1 + np.exp(-x)))
all_data["Value"] = all_data["Value"].apply(lambda x: 1 / (1 + np.exp(-x)))

# split the data into training and testing sets
X = all_data[["Time", "Value"]].values
y = all_data["Code"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# reshape the data to have a shape of (n_samples, 1, 2)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# create a neural network
layers = [
    NeuronLayer(2, 10),  # hidden layer 10 neurons
    NeuronLayer(10, 20),  # output layer 20 neurons
]
nn = NeuralNetwork(layers)

# Initialize a matrix of zeros with shape (n_samples, n_classes)
y_train_encoded = np.zeros((y_train.size, y_train.max() + 1))

# Set the appropriate column for each row to 1 according to the class label
y_train_encoded[np.arange(y_train.size), y_train] = 1

# train the model
nn.train(X_train, y_train_encoded, n_epochs=100, alfa=0.1)

# test the model
y_pred = nn.predict(X_test)

# Convert the raw outputs to class labels
y_pred = np.argmax(y_pred, axis=1)

# Print the results
print("Precision: %.2f" % precision_score(y_test, y_pred))
print("Recall: %.2f" % recall_score(y_test, y_pred))

# Plot the training error
plt.plot(nn.errors)
plt.title(f"error(epoch) learn_rate={0.1}")
plt.xlabel("epoch")
plt.ylabel("error")
plt.show()
