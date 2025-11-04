import ptychi.image_proc as ip
from ptychi.data_structures.object import PlanarObject
from ptychi.data_structures.probe_positions import ProbePositions
from ptychi.api.options.base import ObjectOptions, ProbePositionOptions
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


def main():
    positions = prepare_test_data()[-1]
    positions_original = positions.copy()
    positions = positions + 256 + 0.5
    
    intensity_pred = tifffile.imread("predictions/intensity_pred.tiff")
    phase_pred = tifffile.imread("predictions/phase_pred.tiff")
    
    intensity_pred = ndi.zoom(intensity_pred, [1, 4, 4])
    phase_pred = ndi.zoom(phase_pred, [1, 4, 4])
    
    intensity_pred = intensity_pred[:, 32:96, 32:96]
    phase_pred = phase_pred[:, 32:96, 32:96]
    
    buffer_size = (512, 512)
    stitched_intensity = stitch_patches(torch.tensor(intensity_pred).float(), torch.tensor(positions).float(), buffer_size)
    stitched_phase = stitch_patches(torch.tensor(phase_pred).float(), torch.tensor(positions).float(), buffer_size)
    stitched_intensity = stitched_intensity.numpy()
    stitched_phase = stitched_phase.numpy()
    
    # Get ROI
    ptychi_positions = ProbePositions(
        data=positions_original,
        options=ProbePositionOptions()
    )
    
    dummy_object = PlanarObject(
        data=(torch.from_numpy(stitched_intensity) * torch.exp(1j * torch.from_numpy(stitched_phase)))[None, ...],
        options=ObjectOptions()
    )
    dummy_object.build_roi_bounding_box(ptychi_positions)
    object_roi = dummy_object.get_object_in_roi().detach().cpu().numpy()
    
    stitched_intensity_roi = np.abs(object_roi[0, ...])
    stitched_phase_roi = np.angle(object_roi[0, ...])
    
    tifffile.imwrite("predictions/stitched_intensity.tiff", stitched_intensity)
    tifffile.imwrite("predictions/stitched_phase.tiff", stitched_phase)
    tifffile.imwrite("predictions/stitched_intensity_roi.tiff", stitched_intensity_roi)
    tifffile.imwrite("predictions/stitched_phase_roi.tiff", stitched_phase_roi)


if __name__ == "__main__":
    main()
