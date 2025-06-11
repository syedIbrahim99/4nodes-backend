import pandas as pd

# Read the data from the text file
with open('feb18L10_Nsurface.txt', 'r') as file:
    lines = file.readlines()

# Strip whitespace and split lines by comma to create columns
data = [line.strip().split(',') for line in lines]

# Create a DataFrame with each line as a separate column
df = pd.DataFrame(data)

# Transpose the DataFrame to have each line as a row
df = df.transpose()

# Rename the columns
df.columns = ['N1', 'N2', 'N3', 'N4']

# Save DataFrame to CSV file
df.to_csv('surface_4.csv', index=False)
