import os
import argparse
import numpy as np
import scipy.io as scio
import math
from skimage.metrics import structural_similarity as ssim

def getPSNR(X, Y, max):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        X (np.ndarray): Ground truth image.
        Y (np.ndarray): Denoised/generated image.
        max (float): Maximum possible pixel value in the image.
    
    Returns:
        float: PSNR value.
    """
    ref_data = np.array(X, dtype=np.float32)
    gen_data = np.array(Y, dtype=np.float32)
    diff = ref_data - gen_data
    diff = diff.flatten()
    mse = np.mean(diff ** 2)
    if mse == 0: return 100
    else: return 10 * math.log10(max ** 2 / mse)


def SSIMs_PSNRs(gtr_dir, gen_dir, matfile_dir):
    """
    Calculate SSIM and PSNR for each subject in the test dataset.

    Args:
        gtr_dir (str): Path to ground truth directory containing normal-dose PET (.npz files), 
                       one per subject, named as <subject_name>.npz.
        gen_dir (str): Path to generated/denoised results directory. Each subject has a subfolder 
                       containing a .npz file with the denoised result.
        matfile_dir (str): Path to original 3D PET files used to extract max SUV value per subject.

    Returns:
        Tuple of np.ndarray: (SSIM scores, PSNR scores)
    """
    ssims, psnrs = [], []
    subfolder_list = [d for d in os.listdir(gen_dir) if os.path.isdir(os.path.join(gen_dir, d))]
    print("Subject\tSSIM\tPSNR\t")

    for i in subfolder_list:
        folder_path = os.path.join(gen_dir, i)
        fn = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.npz')]

        if len(fn)>0:
            gen_file_path = os.path.join(folder_path, fn[0])

            # -----------------------------------------------------
            # During preprocessing, both normal-dose and low-dose PET scans were:
            # 1. Converted to SUV units.
            # 2. Clipped such that any voxel value greater than 16 was set to 16.
            # 3. Normalized to [0, 1] by dividing by 16.
            
            # Load generated denoised result and rescale by 16 (inverse of normalization during preprocessing)
            gen = np.load(gen_file_path)["arr_0"] * 16
            gen[gen<0] = 0
            gen = gen.astype(np.float32)

            # Load corresponding ground truth and rescale by 16
            gtr = np.load(os.path.join(gtr_dir, i + '.npz'))['arr_0'][1] * 16

            # Load original 3D PET file to get the maximum SUV value
            matfile_data = scio.loadmat(os.path.join(matfile_dir, i + ".mat"))
            pet_SUV = matfile_data['suv']
            max_value = pet_SUV.max()

            # Calculate SSIM and PSNR
            ssim_value = ssim(gtr, gen, data_range=16)
            psnr_value = getPSNR(gtr, gen, max_value)

            print("{0}\t{1}\t{2}\t".format(i.split(".")[0], ssim_value, psnr_value))

            ssims.append(ssim_value)
            psnrs.append(psnr_value)

    return np.array(ssims), np.array(psnrs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate PSNR and SSIM for denoised PET images.')
    
    parser.add_argument(
        '--gtr_dir', 
        type=str, 
        help='Directory containing ground truth normal-dose PET (.npz) files. '
             'Each subject should have a file named <subject_name>.npz. '
             'Note: You may need to modify the data loading logic depending on your data structure.'
    )
    
    parser.add_argument(
        '--gen_dir', 
        type=str, 
        help='Directory containing the generated denoised results. '
             'Each subject should have a subfolder under this directory, with a .npz file inside '
             'storing the denoised PET image.'
    )

    parser.add_argument(
        '--matfile_dir', 
        type=str, 
        help='Directory containing the original 3D PET files used to retrieve the SUV image. '
             'This is needed to extract the maximum SUV value for accurate PSNR computation. '
             'You may need to adapt the data loading depending on your file format.'
    )

    args = parser.parse_args()

    # Compute SSIM and PSNR across the dataset
    SSIM_measures, PSNR_measures = SSIMs_PSNRs(args.gtr_dir, args.gen_dir, args.matfile_dir)
    print ("SSIM on {0} samples".format(len(SSIM_measures)))
    print ("Mean: {0}\nstd: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

    print ("PSNR on {0} samples".format(len(PSNR_measures)))
    print ("Mean: {0}\nstd: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))


