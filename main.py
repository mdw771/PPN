import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from models.ppn import build_ppn_model
from train.train_ppn import train_ppn
from utils.visualization import process_and_visualize
from utils.data_loader import normalize_data
from config import Config
import scipy.ndimage as ndi
from cone_dataloader import prepare_training_data, prepare_test_data

import h5py
import tifffile


def configure_single_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU detected, running on CPU.")
        return

    primary_gpu = gpus[0]
    try:
        tf.config.set_visible_devices(primary_gpu, 'GPU')
        tf.config.experimental.set_memory_growth(primary_gpu, True)
        print(f"Using GPU: {primary_gpu.name}")
    except RuntimeError as err:
        print(f"Failed to configure GPU settings: {err}")


def main():
    # Set random seed
    tf.keras.backend.clear_session()
    np.random.seed(123)
    tf.random.set_seed(123)

    configure_single_gpu()

    (
        X_train,
        Y_I_train,
        Y_phi_train,
        X_test,
        Y_I_test,
        Y_phi_test,
    ) = prepare_training_data()

    # Build and compile model
    model = build_ppn_model(
        h=Config.IMAGE_HEIGHT,
        w=Config.IMAGE_WIDTH,
        patch_size=Config.PATCH_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        num_heads=Config.NUM_HEADS,
        transformer_layers=Config.TRANSFORMER_LAYERS
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )

    # Train model
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
    
    X_test, positions_test = prepare_test_data()
    predictions = model.predict(X_test)
    intensity_pred, phase_pred = predictions[:2]
    tifffile.imwrite("predictions/intensity_pred.tiff", intensity_pred[..., 0])
    tifffile.imwrite("predictions/phase_pred.tiff", phase_pred[..., 0])

    # # Persist trained model for future inference
    # os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # save_dir = os.path.join(Config.MODEL_SAVE_DIR, timestamp)
    # os.makedirs(save_dir, exist_ok=True)
    # model_path = os.path.join(save_dir, 'saved_ppn_model.keras')
    # model.save(model_path)
    # print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
