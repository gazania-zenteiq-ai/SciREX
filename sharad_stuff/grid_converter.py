import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import sys
import time

def convert_grid(input_file, output_file, res_r, res_theta):
    print(f"Loading data from {input_file}...")
    try:
        # Assuming CSV format. Adjust if it's space-separated (e.g., delim_whitespace=True)
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    # Validate that required columns are present
    expected_cols = ['x', 'y', 'z', 'Ax', 'Ay', 'Az', 'r', 'theta']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the data: {missing_cols}")
        print("Please ensure your CSV has headers: x, y, z, Ax, Ay, Az, r, theta")
        sys.exit(1)

    print(f"Data successfully loaded. Shape: {df.shape}")

    # Extract coordinates defining the non-uniform grid
    coords = df[['r', 'theta']].values

    # Extract the features to be interpolated
    feature_cols = ['x', 'y', 'z', 'Ax', 'Ay', 'Az']
    features = df[feature_cols].values

    # Determine boundaries of the grid
    r_min, r_max = df['r'].min(), df['r'].max()
    theta_min, theta_max = df['theta'].min(), df['theta'].max()

    print(f"r boundary:     [{r_min:.4f}, {r_max:.4f}]")
    print(f"theta boundary: [{theta_min:.4f}, {theta_max:.4f}]")

    # Generate the new uniform grid
    r_uniform = np.linspace(r_min, r_max, res_r)
    theta_uniform = np.linspace(theta_min, theta_max, res_theta)

    # Meshgrid creates the 2D grid combinations
    R_grid, THETA_grid = np.meshgrid(r_uniform, theta_uniform)
    
    # Flatten grid coordinates to a (N, 2) array for the interpolator
    grid_coords = np.column_stack((R_grid.ravel(), THETA_grid.ravel()))

    print(f"\nConstructing new uniform grid with resolution: {res_r} (r) x {res_theta} (theta)")
    print(f"Total grid points to interpolate: {len(grid_coords)}")

    # Initialize Radial Basis Function (RBF) Interpolator
    # Thin-plate spline kernel is used as it minimizes the bending energy of the interpolation 
    # surface. It is considered one of the most mathematically robust methods for scattered 
    # data interpolation (equivalent to Kriging with specific assumptions).
    # Since computational time and memory are not an issue, this is the optimal choice 
    # despite its O(N^3) time and O(N^2) memory complexity.
    print("\nInitializing RBFInterpolator (Thin-Plate Spline)...")
    print("Note: For large N, this will use significant RAM and compute time.")
    start_time = time.time()
    
    try:
        interpolator = RBFInterpolator(
            y=coords, 
            d=features, 
            kernel='thin_plate_spline', 
            smoothing=0.0  # Exact interpolation through data points
        )
    except Exception as e:
        print(f"Error initializing RBFInterpolator: {e}")
        print("If you get a MemoryError, the dataset might be too large for O(N^2) memory.")
        sys.exit(1)

    print(f"Interpolator initialized in {time.time() - start_time:.2f} seconds.")

    # Evaluate the features on the new uniform grid
    print("\nEvaluating interpolated values on the uniform grid...")
    start_time = time.time()
    interpolated_features = interpolator(grid_coords)
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")

    # Assemble the final dataframe
    out_df = pd.DataFrame(interpolated_features, columns=feature_cols)
    out_df['r'] = grid_coords[:, 0]
    out_df['theta'] = grid_coords[:, 1]

    # Reorder columns to exactly match the input structure
    out_df = out_df[expected_cols]

    # Save to the output file
    print(f"\nSaving results to {output_file}...")
    out_df.to_csv(output_file, index=False)
    print("Done!")

    # ==========================================
    # Visualization of the Grid
    # ==========================================
    print("\nGenerating visualization...")
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Original non-uniform points
    plt.subplot(1, 2, 1)
    # Using 'z' (index 2 in features) for color mapping
    plt.scatter(coords[:, 0], coords[:, 1], c=features[:, 2], cmap='viridis', s=15, alpha=0.8)
    plt.title('Original Non-Uniform Grid (Colored by z)')
    plt.xlabel('r')
    plt.ylabel('theta')
    plt.colorbar(label='z')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: New uniform grid
    plt.subplot(1, 2, 2)
    # Reshape interpolated z back to 2D grid for plotting
    z_uniform = interpolated_features[:, 2].reshape(res_theta, res_r)
    
    # Plotting the uniform grid values
    plt.pcolormesh(R_grid, THETA_grid, z_uniform, cmap='viridis', shading='auto')
    
    # Overlay a faint grid to make the "uniform grid" very easily seen
    # We only draw a subset of lines if the resolution is high so it doesn't become purely white
    step_r = max(1, res_r // 30)
    step_theta = max(1, res_theta // 30)
    for r_val in r_uniform[::step_r]:
        plt.axvline(x=r_val, color='white', alpha=0.3, linestyle='-', linewidth=0.5)
    for t_val in theta_uniform[::step_theta]:
        plt.axhline(y=t_val, color='white', alpha=0.3, linestyle='-', linewidth=0.5)
        
    plt.title('Interpolated Uniform Grid (Colored by z)')
    plt.xlabel('r')
    plt.ylabel('theta')
    plt.colorbar(label='z')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # ==========================================
    # SET YOUR FILE PATHS AND PARAMETERS HERE
    # ==========================================
    INPUT_FILE = "data/20250804_stator_magnetOD_28_1_Az_30d.txt"     # <-- Change this to your input file
    OUTPUT_FILE = "data/20250804_stator_magnetOD_28_1_Az_30d_uniform.csv"  # <-- Change this to your desired output file
    RESOLUTION_R = 1000
    RESOLUTION_THETA = 1000
    
    convert_grid(INPUT_FILE, OUTPUT_FILE, RESOLUTION_R, RESOLUTION_THETA)
