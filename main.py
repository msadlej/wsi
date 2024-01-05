import pandas as pd

# Initialize an empty dictionary to hold the data
all_data = {}

# Loop over the file names
for i in range(1, 71):
    # Generate the file name
    file_name = f"Diabetes-Data/data-{i:02d}"

    # Read the file into a DataFrame
    df = pd.read_csv(file_name, delimiter="\t", header=None)

    # Assign column names
    df.columns = ["Date", "Time", "Code", "Value"]

    # Store the DataFrame in the dictionary
    all_data[i] = df

# Now all_data is a dictionary where the key is the file number and the value is the DataFrame
# For example, to print the DataFrame for file 1, you can do:
print(all_data[1])
