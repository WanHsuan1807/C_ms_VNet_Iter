import os
import numpy as np
import nibabel as nib


# -----------------------------
# 讀取 RAW volume
# -----------------------------
def read_raw_volume(raw_path, shape, dtype=np.uint8):

    file_size = os.path.getsize(raw_path)
    expected_size = np.prod(shape) * np.dtype(dtype).itemsize

    header_size = file_size - expected_size

    if header_size < 0:
        raise ValueError("Shape 或 dtype 設錯")

    print("File size:", file_size)
    print("Expected size:", expected_size)
    print("Header size:", header_size)

    volume = np.fromfile(
        raw_path,
        dtype=dtype,
        offset=header_size
    )

    volume = volume.reshape(shape)

    return volume


# -----------------------------
# intensity normalization
# -----------------------------
def normalize_intensity(volume):

    volume = volume.astype(np.float32)

    p1 = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)

    volume = np.clip(volume, p1, p99)

    volume = (volume - volume.min()) / (volume.max() - volume.min())

    return volume


# -----------------------------
# crop background
# -----------------------------
def crop_background(volume, threshold=0.05):

    mask = volume > threshold

    coords = np.argwhere(mask)

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    cropped = volume[z0:z1, y0:y1, x0:x1]

    return cropped


# -----------------------------
# sliding window patch extraction
# -----------------------------
def extract_patches(volume, patch_size=128, stride=64):

    patches = []

    z, y, x = volume.shape

    for zz in range(0, z - patch_size + 1, stride):
        for yy in range(0, y - patch_size + 1, stride):
            for xx in range(0, x - patch_size + 1, stride):

                patch = volume[
                    zz:zz + patch_size,
                    yy:yy + patch_size,
                    xx:xx + patch_size
                ]

                patches.append(patch)

    return patches


# -----------------------------
# save NIfTI
# -----------------------------
def save_nifti(volume, save_path):

    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))

    nib.save(nifti_img, save_path)


# -----------------------------
# 主 pipeline
# -----------------------------
def convert_raw_to_patches(
        raw_path,
        shape,
        output_dir,
        dtype=np.uint8,
        patch_size=128,
        stride=64):

    os.makedirs(output_dir, exist_ok=True)

    print("\nReading RAW volume...")
    volume = read_raw_volume(raw_path, shape, dtype)

    print("Original shape:", volume.shape)

    print("\nNormalizing intensity...")
    volume = normalize_intensity(volume)

    print("\nCropping background...")
    volume = crop_background(volume)

    print("Cropped shape:", volume.shape)

    print("\nExtracting patches...")
    patches = extract_patches(volume, patch_size, stride)

    print("Total patches:", len(patches))

    print("\nSaving patches...")

    for i, patch in enumerate(patches):

        filename = f"patch_{i:05d}.nii.gz"

        save_path = os.path.join(output_dir, filename)

        save_nifti(patch, save_path)

    print("\nDone!")


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":

    raw_file = "20260312_3dvolume/Component_6_2_1_288_1884_732_351.raw"

    # RAW volume shape (Z,Y,X)
    shape = (351, 732, 1884)

    output_dir = "patches_1"

    convert_raw_to_patches(
        raw_file,
        shape,
        output_dir,
        dtype=np.uint8,
        patch_size=256,
        stride=128
    )