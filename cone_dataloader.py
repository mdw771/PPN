import numpy as np
from utils.data_loader import normalize_data
from config import Config
import scipy.ndimage as ndi
import h5py
import tifffile
import tqdm


def process_dp(dp):
    dp = dp - dp.mean(0)
    dp = normalize_data(dp)
    return dp


def prepare_training_data(regenerate=False):
    if regenerate:
        X = []
        Y_I = []
        Y_phi = []
        for angle in tqdm.trange(90, 180, 1):
            with h5py.File(f"../data/cone/angle_{angle}/ptychodus_dp.hdf5", "r") as f:
                dp = f["dp"][...]
                # 128 -> 64
                # dp = dp[:, 32:96, 32:96]
                # 64 -> 32
                # dp = ndi.zoom(dp, [1, 0.5, 0.5])
                dp = process_dp(dp)
                X.append(dp)
                        
            obj_patches_mag = tifffile.imread(f"../data/cone/angle_{angle}/object_patches_mag.tiff")[:, 0]
            obj_patches_phase = tifffile.imread(f"../data/cone/angle_{angle}/object_patches_phase.tiff")[:, 0]

            # obj_patches_mag = ndi.zoom(obj_patches_mag, [1, 0.5, 0.5])
            # obj_patches_phase = ndi.zoom(obj_patches_phase, [1, 0.5, 0.5])
            
            obj_patches_mag = normalize_data(obj_patches_mag)
            obj_patches_phase = normalize_data(obj_patches_phase)

            Y_I.append(obj_patches_mag)
            Y_phi.append(obj_patches_phase)

        X = np.concatenate(X, axis=0)
        Y_I = np.concatenate(Y_I, axis=0)
        Y_phi = np.concatenate(Y_phi, axis=0)
        
        tifffile.imwrite("inputs/X.tiff", X)
        tifffile.imwrite("inputs/Y_I.tiff", Y_I)
        tifffile.imwrite("inputs/Y_phi.tiff", Y_phi)
    else:
        X = tifffile.imread("inputs/X.tiff")
        Y_I = tifffile.imread("inputs/Y_I.tiff")
        Y_phi = tifffile.imread("inputs/Y_phi.tiff")

    X = X.astype(np.float32)
    Y_I = Y_I.astype(np.float32)
    Y_phi = Y_phi.astype(np.float32)

    X = X[..., None]
    Y_I = Y_I[..., None]
    Y_phi = Y_phi[..., None]
    
    print("X.shape, Y_I.shape, Y_phi.shape:", X.shape, Y_I.shape, Y_phi.shape)
    
    train_indices = np.random.choice(len(X), size=int(len(X) * Config.TRAIN_RATIO), replace=False)
    test_indices = np.setdiff1d(np.arange(len(X)).astype(int), train_indices)
    X_train = X[train_indices]
    Y_I_train = Y_I[train_indices]
    Y_phi_train = Y_phi[train_indices]
    X_test = X[test_indices]
    Y_I_test = Y_I[test_indices]
    Y_phi_test = Y_phi[test_indices]
    return (
        X_train,
        Y_I_train,
        Y_phi_train,
        X_test,
        Y_I_test,
        Y_phi_test,
    )


def prepare_test_data():
    with h5py.File("../data/cone/angle_0/ptychodus_dp.hdf5", "r") as f:
        dp = f["dp"][...]
        # 128 -> 64
        # dp = dp[:, 32:96, 32:96]
        # 64 -> 32
        # dp = ndi.zoom(dp, [1, 0.5, 0.5])
        dp = process_dp(dp)
        X = dp[..., None].astype(np.float32)
        
    with h5py.File("../data/cone/angle_0/ptychodus_para.hdf5", "r") as f:
        positions = np.stack([f["probe_position_y_m"][...], f["probe_position_x_m"][...]], axis=1)
        dx = f["object"].attrs["pixel_width_m"]
        positions = positions / dx
    
    return X, positions
