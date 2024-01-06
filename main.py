import pandas as pd

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

# Print the unique 'Code' values
print(unique_codes)

# Create a dictionary that maps each unique 'Code' to an integer from 0 to 19
code_map = {code: i for i, code in enumerate(unique_codes)}

# Replace the 'Code' values in the DataFrame using the dictionary
all_data["Code"] = all_data["Code"].map(code_map)

# Print the DataFrame
print(all_data)


"""
data-20: line 104 -> Code 4
data-27: lines 806-811 -> WTF!?
data-29: MANY WTFS!?
"""
