import tensorflow as tf
import numpy as np
from models.ppn import build_ppn_model
from train.train_ppn import train_ppn
from utils.data_loader import load_and_prepare_data
from utils.visualization import process_and_visualize
from config import Config
import scipy.ndimage as ndi

import h5py
import tifffile


def prepare_data():
    X = []
    Y_I = []
    Y_phi = []
    
    for angle in range(0, 1):
        with h5py.File(f"../data/cone/angle_{angle}/ptychodus_dp.hdf5", "r") as f:
            dp = f["dp"][...]
            # 128 -> 64
            dp = dp[:, 32:96, 32:96]
            # 64 -> 32
            dp = ndi.zoom(dp, [1, 0.5, 0.5])
            X.append(dp)
        
        obj_patches_mag = tifffile.imread(f"../data/cone/angle_{angle}/object_patches_mag.tiff")[:, 0]
        obj_patches_phase = tifffile.imread(f"../data/cone/angle_{angle}/object_patches_phase.tiff")[:, 0]
        
        obj_patches_mag = ndi.zoom(obj_patches_mag, [1, 0.25, 0.25])
        obj_patches_phase = ndi.zoom(obj_patches_phase, [1, 0.25, 0.25])
        
        Y_I.append(obj_patches_mag)
        Y_phi.append(obj_patches_phase)

    X = np.concatenate(X, axis=0)[..., None]
    Y_I = np.concatenate(Y_I, axis=0)[..., None]
    Y_phi = np.concatenate(Y_phi, axis=0)[..., None]
    train_indices = np.random.choice(len(X), size=int(len(X) * Config.TRAIN_RATIO), replace=False)
    test_indices = np.setdiff1d(np.arange(len(X)).astype(int), train_indices)
    X_train = X[train_indices]
    Y_I_train = Y_I[train_indices]
    Y_phi_train = Y_phi[train_indices]
    X_test = X[test_indices]
    Y_I_test = Y_I[test_indices]
    Y_phi_test = Y_phi[test_indices]
    return X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test


def main():
    # Set random seed
    tf.keras.backend.clear_session()
    np.random.seed(123)
    tf.random.set_seed(123)

    if False:
        # Load and prepare data
        X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test = load_and_prepare_data(
            Config.AMPLITUDE_PATH,
            Config.PHASE_PATH,
            Config.PROBE_PATH,
            new_size=Config.IMAGE_SIZE,
            overlap_rate=Config.OVERLAP_RATE,
            ratio=Config.TRAIN_RATIO
        )

        # Load sub-images for visualization
        amplitude = load_and_preprocess_image(Config.AMPLITUDE_PATH, Config.IMAGE_SIZE)
        phase = load_and_preprocess_image(Config.PHASE_PATH, Config.IMAGE_SIZE)
        amplitude, phase = adjust_amplitude_phase(amplitude, phase)
        sub_image1 = amplitude[412:512, 412:512]
        sub_image2 = phase[412:512, 412:512]
        
    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test = prepare_data()

    # Build model
    model = build_ppn_model(
        h=Config.IMAGE_HEIGHT,
        w=Config.IMAGE_WIDTH,
        patch_size=Config.PATCH_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        num_heads=Config.NUM_HEADS,
        transformer_layers=Config.TRANSFORMER_LAYERS
    )

    # Train model
    print(X_train.shape)
    print(Y_I_train.shape)
    print(Y_phi_train.shape)
    print(X_test.shape)
    print(Y_I_test.shape)
    print(Y_phi_test.shape)
    history, predictions = train_ppn(
        model,
        X_train,
        Y_I_train,
        Y_phi_train,
        X_test,
        Y_I_test,
        Y_phi_test,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS
    )

    # Visualize results
    process_and_visualize(
        model,
        X_test,
        Y_I_test,
        Y_phi_test,
        point_size=Config.POINT_SIZE,
        overlap=Config.OVERLAP,
        sub_image1=None,
        sub_image2=None
    )

if __name__ == "__main__":
    main()