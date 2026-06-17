"""
Preprocesses raw Car Design ShapeNet data into x.npy, y.npy, surf.npy
matching the reference Transolver repo exactly.

Reference: Transolver/Car-Design-ShapeNetCar/dataset/dataset.py

Input per sample:
  hexvelo_smpl.vtk   - volume mesh, point_vectors = (vx, vy, vz)
  quadpress_smpl.vtk - surface mesh, point_scalars = pressure

Output per sample (written in-place):
  x.npy    - (N, 7)  [x, y, z, sdf, nx, ny, nz]
               exterior points first, then surface points
  y.npy    - (N, 4)  [vx, vy, vz, pressure]
  surf.npy - (N,)    bool, False for exterior / True for surface

Point ordering: N = len(exterior) + len(surface)
  - exterior: volume mesh points NOT on the surface
  - surface:  quad mesh points (pressure mesh)
"""

import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.neighbors import NearestNeighbors

TRAINING_DATA_DIR = "/home/harshdeep/Harshdeep/Data/mlcfd_data/training_data"


def load_vtk_unstructured(path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def get_sdf(target, boundary):
    """Distance + unit direction from each target point to nearest boundary point."""
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(grid):
    """Compute per-point surface normals from an unstructured grid (matches reference)."""
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(grid)
    surface_filter.Update()
    poly_data = surface_filter.GetOutput()

    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()

    grid.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(grid)
    c2p.Update()

    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.float64)
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)

    if np.isnan(normal).any():
        return get_normal(grid)
    return normal


def preprocess_sample(sample_path):
    hex_path  = os.path.join(sample_path, "hexvelo_smpl.vtk")
    quad_path = os.path.join(sample_path, "quadpress_smpl.vtk")

    if not os.path.exists(hex_path) or not os.path.exists(quad_path):
        return False

    grid_velo  = load_vtk_unstructured(hex_path)
    grid_press = load_vtk_unstructured(quad_path)

    points_velo  = vtk_to_numpy(grid_velo.GetPoints().GetData()).astype(np.float64)
    velo         = vtk_to_numpy(grid_velo.GetPointData().GetArray("point_vectors")).astype(np.float64)
    points_press = vtk_to_numpy(grid_press.GetPoints().GetData()).astype(np.float64)
    press        = vtk_to_numpy(grid_press.GetPointData().GetArray("point_scalars")).astype(np.float64)

    # Separate exterior (volume − surface) from surface points
    surface_set      = {tuple(p) for p in points_press}
    exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface_set]
    velo_dict        = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

    pos_ext  = points_velo[exterior_indices]
    pos_surf = points_press

    # SDF + direction-to-surface for exterior; SDF=0 + actual normals for surface
    sdf_ext,  normal_ext  = get_sdf(pos_ext, points_press)
    sdf_surf              = np.zeros(pos_surf.shape[0])
    normal_surf           = get_normal(grid_press)

    # Velocities: from volume mesh for both; pressure=0 for exterior
    velo_ext  = velo[exterior_indices]
    velo_surf = np.array([
        velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3)
        for p in pos_surf
    ])
    press_ext = np.zeros(len(exterior_indices))

    # Build input (N, 7) and target (N, 4)
    init_ext  = np.c_[pos_ext,  sdf_ext,  normal_ext]   # (ext, 7)
    init_surf = np.c_[pos_surf, sdf_surf, normal_surf]   # (surf, 7)
    tgt_ext   = np.c_[velo_ext,  press_ext]              # (ext, 4)
    tgt_surf  = np.c_[velo_surf, press]                  # (surf, 4)

    x    = np.concatenate([init_ext,  init_surf],  axis=0).astype(np.float32)
    y    = np.concatenate([tgt_ext,   tgt_surf],   axis=0).astype(np.float32)
    surf = np.concatenate([
        np.zeros(len(pos_ext),  dtype=bool),
        np.ones(len(pos_surf),  dtype=bool),
    ])

    np.save(os.path.join(sample_path, "x.npy"),    x)
    np.save(os.path.join(sample_path, "y.npy"),    y)
    np.save(os.path.join(sample_path, "surf.npy"), surf)
    return True


def main():
    param_dirs = sorted([
        d for d in os.listdir(TRAINING_DATA_DIR)
        if os.path.isdir(os.path.join(TRAINING_DATA_DIR, d))
    ])

    total, done, skipped = 0, 0, 0
    for param in param_dirs:
        param_path = os.path.join(TRAINING_DATA_DIR, param)
        samples = sorted(os.listdir(param_path))
        print(f"\n[{param}] — {len(samples)} samples")
        for sample in samples:
            sample_path = os.path.join(param_path, sample)
            if not os.path.isdir(sample_path):
                continue
            total += 1
            # Skip only if already preprocessed in the new 7-feature format
            x_path = os.path.join(sample_path, "x.npy")
            if os.path.exists(x_path) and np.load(x_path).shape[1] == 7:
                skipped += 1
                continue
            ok = preprocess_sample(sample_path)
            if ok:
                done += 1
            if (done + skipped) % 50 == 0 and (done + skipped) > 0:
                print(f"  processed {done}, skipped {skipped}/{total}")

    print(f"\nDone. Processed: {done}, Skipped (already correct): {skipped}, Total: {total}")


if __name__ == "__main__":
    main()
