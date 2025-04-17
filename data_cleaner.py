import os
import nibabel as nib
import matplotlib.pyplot as plt
import shutil
import random
import numpy as np
from PIL import Image

def cleanup_brats_dataset(dataset_root):
    """
    Removes all files except *_t1ce.nii.gz and *_seg.nii.gz
    from each subject folder in the BraTS dataset.
    
    Parameters:
    - dataset_root (str): Path to the folder containing subject folders.
    """
    keep_suffixes = ['_t1ce.nii.gz', '_seg.nii.gz']

    for subject in os.listdir(dataset_root):
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        for file in os.listdir(subject_path):
            if not any(file.endswith(suffix) for suffix in keep_suffixes):
                file_path = os.path.join(subject_path, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    
    print(" Cleanup complete. Only t1ce and seg files remain.")

#cleanup_brats_dataset('./dataset/BraTS2021_Extracted')
def split_subjects(src_root, dst_root, train_ratio=0.8, seed=42):
    random.seed(seed)
    subjects = sorted([d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))])
    random.shuffle(subjects)

    split_idx = int(len(subjects) * train_ratio)
    train_subjects = subjects[:split_idx]
    test_subjects = subjects[split_idx:]

    for subset, sub_list in [('train', train_subjects), ('test', test_subjects)]:
        for subject in sub_list:
            src_path = os.path.join(src_root, subject)
            dst_path = os.path.join(dst_root, subset, subject)
            os.makedirs(dst_path, exist_ok=True)
            for file in os.listdir(src_path):
                shutil.copy(os.path.join(src_path, file), os.path.join(dst_path, file))

    print(f" Split complete: {len(train_subjects)} train, {len(test_subjects)} test.")

# split_subjects(
#     src_root='./dataset/BraTS2021_Extracted',
#     dst_root='./dataset/BraTS2021_Split',
#     train_ratio=0.8
# )





def save_core_slices(subject_root):
    for subset in ['train', 'test']:
        subset_path = os.path.join(subject_root, subset)
        for subject in os.listdir(subset_path):
            subj_path = os.path.join(subset_path, subject)
            t1ce_file = os.path.join(subj_path, f'{subject}_t1ce.nii.gz')
            seg_file = os.path.join(subj_path, f'{subject}_seg.nii.gz')

            if not (os.path.exists(t1ce_file) and os.path.exists(seg_file)):
                print(f"Deleting {subject} (missing t1ce or seg)")
                shutil.rmtree(subj_path, ignore_errors=True)
                continue

            try:
                t1ce = nib.load(t1ce_file).get_fdata()
                seg = nib.load(seg_file).get_fdata()
            except Exception as e:
                print(f"Error loading {subject}: {e}")
                shutil.rmtree(subj_path, ignore_errors=True)
                continue

            t1ce_out = os.path.join(subj_path, 't1ce')
            seg_out = os.path.join(subj_path, 'seg')
            os.makedirs(t1ce_out, exist_ok=True)
            os.makedirs(seg_out, exist_ok=True)

            slice_saved = False

            for z in range(seg.shape[2]):
                core_mask = (seg[:, :, z] == 1) | (seg[:, :, z] == 4)
                if not np.any(core_mask):
                    continue

                img_slice = t1ce[:, :, z]
                if np.all(img_slice == img_slice.flat[0]):
                    continue

                img_norm = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
                img_uint8 = (img_norm * 255).astype(np.uint8)
                mask_uint8 = core_mask.astype(np.uint8) * 255

                Image.fromarray(img_uint8).save(os.path.join(t1ce_out, f'slice{z:03}.png'))
                Image.fromarray(mask_uint8).save(os.path.join(seg_out, f'slice{z:03}.png'))
                slice_saved = True

            # 🗑️ If no useful slice was saved, remove the subject
            if not slice_saved:
                print(f"🗑️ Deleting {subject} (no valid slices)")
                shutil.rmtree(subj_path, ignore_errors=True)

    print("Completed filtering and saving valid slices.")

#save_core_slices('./dataset/BraTS2021_Split')

def delete_nii_gz_files(subject_root):
    """
    Deletes all .nii.gz files from each subject in train/ and test/ folders.
    Keeps only the processed slice PNGs.
    """
    for subset in ['train', 'test']:
        subset_path = os.path.join(subject_root, subset)
        for subject in os.listdir(subset_path):
            subj_path = os.path.join(subset_path, subject)
            if not os.path.isdir(subj_path):
                continue

            for file in os.listdir(subj_path):
                if file.endswith('.nii.gz'):
                    file_path = os.path.join(subj_path, file)
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {file_path}")
    
    print("All .nii.gz files deleted.")

#delete_nii_gz_files('./dataset/BraTS2021_Split')
