import ptychi.image_proc as ip
import numpy as np
import torch
import tifffile
import scipy.ndimage as ndi

from cone_dataloader import prepare_test_data


def stitch_patches(patches: np.ndarray, positions: np.ndarray, buffer_size: tuple[int, int]) -> np.ndarray:
    buffer = torch.zeros(buffer_size)
    occupancy = torch.zeros(buffer_size)
    buffer = ip.place_patches_fourier_shift(buffer, positions, patches, "add", False, pad=1)
    occupancy = ip.place_patches_fourier_shift(occupancy, positions, torch.ones_like(patches), "add", False, pad=1)
    return buffer / occupancy


def main(original_size: int = 32):
    positions = prepare_test_data()[-1]
    positions = positions + 256
    
    intensity_pred = tifffile.imread("predictions/intensity_pred.tiff")
    phase_pred = tifffile.imread("predictions/phase_pred.tiff")
    
    intensity_pred = ndi.zoom(intensity_pred, [1, 128 / original_size, 128 / original_size])
    phase_pred = ndi.zoom(phase_pred, [1, 128 / original_size, 128 / original_size])
    
    intensity_pred = intensity_pred[:, 32:96, 32:96]
    phase_pred = phase_pred[:, 32:96, 32:96]
    
    buffer_size = (512, 512)
    stitched_intensity = stitch_patches(torch.tensor(intensity_pred).float(), torch.tensor(positions).float(), buffer_size)
    stitched_phase = stitch_patches(torch.tensor(phase_pred).float(), torch.tensor(positions).float(), buffer_size)
    stitched_intensity = stitched_intensity.numpy()
    stitched_phase = stitched_phase.numpy()
    tifffile.imwrite("predictions/stitched_intensity.tiff", stitched_intensity)
    tifffile.imwrite("predictions/stitched_phase.tiff", stitched_phase)


if __name__ == "__main__":
    main(original_size=64)
