import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import sys
import time
import os

def convert_grid(input_file, output_file, res_r, res_theta):
    print(f"Loading data from {input_file}...")
    try:
        # Load the data using whitespace separator as the input files are tab/space separated
        df = pd.read_csv(input_file, sep='\s+')
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    # Validate that required columns are present
    expected_cols = ['x', 'y', 'z', 'Ax', 'Ay', 'Az']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the data: {missing_cols}")
        print("Please ensure your data has columns: x, y, z, Ax, Ay, Az")
        sys.exit(1)

    # Calculate polar coordinates if they don't exist
    if 'r' not in df.columns or 'theta' not in df.columns:
        print("Calculating r and theta from x and y...")
        df['r'] = np.sqrt(df['x']**2 + df['y']**2)
        df['theta'] = np.arctan2(df['y'], df['x'])

    # Drop duplicates to avoid Singular Matrix error in RBFInterpolator
    initial_count = len(df)
    df.drop_duplicates(subset=['r', 'theta'], inplace=True)
    final_count = len(df)
    if initial_count != final_count:
        print(f"Dropped {initial_count - final_count} duplicate points.")

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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False)
    print("Done!")

    # ==========================================
    # Visualization of the Grid
    # ==========================================
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- ROW 1: Z Coordinate ---
    # Plot 1,1: Original non-uniform points (z)
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=features[:, 2], cmap='viridis', s=15, alpha=0.8)
    ax1.set_title('Original Non-Uniform Grid (z)')
    ax1.set_xlabel('r')
    ax1.set_ylabel('theta')
    fig.colorbar(sc1, ax=ax1, label='z')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 1,2: New uniform grid (z)
    ax2 = axes[0, 1]
    z_uniform = interpolated_features[:, 2].reshape(res_theta, res_r)
    pcm1 = ax2.pcolormesh(R_grid, THETA_grid, z_uniform, cmap='viridis', shading='auto')
    ax2.set_title('Interpolated Uniform Grid (z)')
    ax2.set_xlabel('r')
    ax2.set_ylabel('theta')
    fig.colorbar(pcm1, ax=ax2, label='z')

    # Overlay grid lines on ax2
    step_r = max(1, res_r // 30)
    step_theta = max(1, res_theta // 30)
    for r_val in r_uniform[::step_r]:
        ax2.axvline(x=r_val, color='white', alpha=0.2, linestyle='-', linewidth=0.5)
    for t_val in theta_uniform[::step_theta]:
        ax2.axhline(y=t_val, color='white', alpha=0.2, linestyle='-', linewidth=0.5)

    # --- ROW 2: Az Field ---
    # Plot 2,1: Original non-uniform points (Az)
    ax3 = axes[1, 0]
    sc2 = ax3.scatter(coords[:, 0], coords[:, 1], c=features[:, 5], cmap='inferno', s=15, alpha=0.8)
    ax3.set_title('Original Non-Uniform Grid (Az)')
    ax3.set_xlabel('r')
    ax3.set_ylabel('theta')
    fig.colorbar(sc2, ax=ax3, label='Az')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Plot 2,2: New uniform grid (Az)
    ax4 = axes[1, 1]
    az_uniform = interpolated_features[:, 5].reshape(res_theta, res_r)
    pcm2 = ax4.pcolormesh(R_grid, THETA_grid, az_uniform, cmap='inferno', shading='auto')
    ax4.set_title('Interpolated Uniform Grid (Az)')
    ax4.set_xlabel('r')
    ax4.set_ylabel('theta')
    fig.colorbar(pcm2, ax=ax4, label='Az')

    # Overlay grid lines on ax4
    for r_val in r_uniform[::step_r]:
        ax4.axvline(x=r_val, color='white', alpha=0.2, linestyle='-', linewidth=0.5)
    for t_val in theta_uniform[::step_theta]:
        ax4.axhline(y=t_val, color='white', alpha=0.2, linestyle='-', linewidth=0.5)
        
    plt.tight_layout()
    
    # Save the plot in the same directory as the output file
    plot_path = os.path.splitext(output_file)[0] + "_comparison.png"
    print(f"Saving visualization to {plot_path}...")
    plt.savefig(plot_path, dpi=300)
    
    plt.show()

if __name__ == '__main__':
    # ==========================================
    # SET YOUR FILE PATHS AND PARAMETERS HERE
    # ==========================================
    INPUT_FILE = "Project_data/varying_diameter/20250820_stator_magnetOD_34_1_Az_30d.txt"     # <-- Change this to your input file
    OUTPUT_FILE = "Project_data/varying_diameter/uniform/20250820_stator_magnetOD_34_1_Az_30d_uniform.csv"  # <-- Change this to your desired output file
    RESOLUTION_R = 128
    RESOLUTION_THETA = 128
    
    convert_grid(INPUT_FILE, OUTPUT_FILE, RESOLUTION_R, RESOLUTION_THETA)
