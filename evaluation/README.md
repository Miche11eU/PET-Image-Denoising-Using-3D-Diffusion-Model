## Usage

- `--gtr_dir`: Path to the folder containing ground-truth **normal-dose** PET images in `.npz` format.  
  Each subject should have a file named `<subject_id>.npz`.  
  *Modify the file loading logic if your ground truth structure differs.*

- `--gen_dir`: Path to the folder containing **generated denoised results**.  
  Each subject should have a subfolder under this directory, and a `.npz` file within that contains the denoised PET image.

- `--matfile_dir`: Path to the folder containing **original PET files without preprocessing** for each subject.  
  These are used to extract the maximum SUV value for PSNR calculation.  
  *Adapt the file loading code as needed based on your structure.*

### Preprocessing Note

During data preprocessing, all PET images were:

1. Converted to SUV units  
2. Clipped to a maximum value of 16  
3. Normalized to the range 0–1 by dividing by 16  

The evaluation script multiplies both predictions and ground truth by 16 before computing the metrics.

### Output

The script prints per-subject SSIM and PSNR values, as well as their mean and standard deviation across the dataset.
