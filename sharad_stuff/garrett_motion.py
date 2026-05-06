import pandas as pd
from grid_converter import convert_grid
import numpy as np

input_file_path = "data/20250804_stator_magnetOD_28_1_Az_30d.txt"    
output_file_path = "data/20250804_stator_magnetOD_28_1_Az_30d_uniform.csv" 
resolution_r = 500
resolution_theta = 500

print(f"Loading raw data from {input_file_path}...")
# The raw file is space/tab delimited
df = pd.read_csv(input_file_path, delim_whitespace=True)

# Calculate the polar coordinates
df['r'] = np.sqrt(df['x']**2 + df['y']**2)
df['theta'] = np.arctan2(df['y'], df['x'])

# grid_converter expects a CSV file containing all 8 columns (including r and theta)
temp_csv_file = "data/temp_processed_data.csv"
print(f"Saving processed data to {temp_csv_file} for conversion...")
df.to_csv(temp_csv_file, index=False)

# Pass the temporary CSV to convert_grid
convert_grid(temp_csv_file, output_file_path, resolution_r, resolution_theta)
