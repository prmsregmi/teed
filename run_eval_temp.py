
import os
from eval import generate_mat_files, call_ods_ois


    # Get all folders starting with "checkpoints" in current directory
checkpoint_dirs = [d for d in os.listdir("result/") if d.startswith("checkpoints") and os.path.isdir(d)]

# Loop through each checkpoints folder
# for checkpoint_dir in checkpoint_dirs:        
for i in range(1, 15, 2):
    img_dir = f"result/checkpoints/CLASSIC/{i}/{i}_model.pth/fused/"
    # mat_dir = generate_mat_files(img_dir)
    call_ods_ois("Classic", img_dir + 'result_mat')