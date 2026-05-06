import os
import glob
import pandas as pd
import numpy as np
import sys
import argparse

# Ensure python can find grid_converter since it is in the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grid_converter import convert_grid

# Disable matplotlib's plt.show() so the script doesn't pause waiting for you to close windows
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: print("(Plotting window disabled during conversion)")

def process_single_file(input_file, res_r=500, res_theta=500, file_id=1):
    """
    Reads a single raw file, computes r and theta, and creates a uniform grid CSV.
    """
    print(f"\n{'='*60}")
    print(f"Processing File: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
        
    data_dir = os.path.dirname(input_file)
    if not data_dir:
        data_dir = "."
        
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(data_dir, f"{base_name}_uniform.csv")
    
    # Skip if the uniform file already exists
    if os.path.exists(output_file):
        print(f"Skipping {input_file}: {output_file} already exists.")
        return
        
    # 1. Read the data
    try:
        df = pd.read_csv(input_file)
        if 'Az' not in df.columns:
            # Fallback to whitespace separated
            df = pd.read_csv(input_file, sep=r'\s+')
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
        
    # 2. Calculate r and theta if not present
    if 'r' not in df.columns and 'x' in df.columns and 'y' in df.columns:
        df['r'] = np.sqrt(df['x']**2 + df['y']**2)
        df['theta'] = np.arctan2(df['y'], df['x'])
    elif 'r' not in df.columns:
        print(f"Skipping {input_file}: Missing 'x' and 'y' columns needed to compute polar coordinates.")
        return
        
    # 3. Drop duplicate coordinates to prevent RBF "Singular Matrix" errors
    initial_len = len(df)
    df = df.drop_duplicates(subset=['x', 'y'])
    if len(df) < initial_len:
        print(f"  Dropped {initial_len - len(df)} duplicate points to prevent singular matrix error.")
        
    # 4. Save to a unique temporary file to prevent conflicts
    temp_file = os.path.join(data_dir, f"temp_{base_name}_processed.csv")
    df.to_csv(temp_file, index=False)
    
    # 5. Call convert_grid
    try:
        convert_grid(temp_file, output_file, res_r, res_theta)
        
        # 6. Add the distinguishing file_id feature to the newly created uniform CSV
        if os.path.exists(output_file):
            out_df = pd.read_csv(output_file)
            out_df['file_id'] = file_id
            out_df.to_csv(output_file, index=False)
            
        print(f"SUCCESS: Output (with file_id={file_id}) saved successfully to --> {output_file}")
    except Exception as e:
        print(f"FAILED to convert {input_file}: {e}")
        
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)

def process_all_files(data_dir="data", res_r=500, res_theta=500):
    """
    Finds all raw text files in a directory and processes them one by one.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return

    # Find all .txt files
    search_path = os.path.join(data_dir, "*.txt")
    files = glob.glob(search_path)
    
    # Filter files
    files_to_process = [f for f in files if not f.endswith("_uniform.csv") and not f.endswith("_uniform.txt")]
    
    if not files_to_process:
        print(f"No valid raw files found to process in '{data_dir}/'")
        return
        
    print(f"Found {len(files_to_process)} files to process.")
    
    # Process them one by one, assigning a unique file_id to each
    for idx, f in enumerate(files_to_process, start=1):
        process_single_file(f, res_r, res_theta, file_id=idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate raw non-uniform TXT scattered points to a uniform grid CSV.")
    parser.add_argument("-i", "--input", type=str, default=None, 
                        help="Optional: Process a single file only (e.g., -i data/my_file.txt)")
    parser.add_argument("-d", "--dir", type=str, default="data", 
                        help="Optional: Directory to process all .txt files if no input file is specified (default: 'data')")
    args = parser.parse_args()
    
    if args.input:
        # Run explicitly on one file
        process_single_file(args.input)
    else:
        # Run on the whole directory automatically
        process_all_files(data_dir=args.dir)
