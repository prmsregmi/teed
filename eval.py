import os
import cv2
import sys
import numpy as np
from scipy.io import savemat


import subprocess

def run_ods_ois(model):
    # Path to the external Python interpreter if different from the main project's
    external_python = ".venv/bin/python3"
    
    # Build the command
    cmd = [
        external_python,
        "main.py",
        "--alg", "TEED " + model,
        "--model_name_list", model,
        "--result_dir", "../../result/result_mat",
        "--save_dir", "../../result/eval_output",
        "--gt_dir", "../../result/gt_mat",
        "--key", "result",
        "--file_format", ".mat",
        "--workers", "-1"
    ]
    
    # Call the script, changing to its directory
    subprocess.run(cmd, cwd="external/edge_eval")

def generate_mat_files(model):
    # === Paths ===
    image_dir = os.path.join("result", "BIPED2UDED", model)
    mat_dir = "result/result_mat"
    os.makedirs(mat_dir, exist_ok=True)
    for i, file in enumerate(os.listdir(image_dir)):
        if file.endswith(".png") or file.endswith(".jpg"):
            img = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_GRAYSCALE) 
            img = img.astype(np.float32) / 255.0  # Normalize if needed
            save_path = os.path.join(mat_dir, os.path.splitext(file)[0] + ".mat")
            savemat(save_path, {"result": img})
            print(f"File {i}, {file} saved as mat.")
    
    run_ods_ois(model)

    ## Code for converting GT to mat

    # gt_dir = "data/BIPED/BIPED/edges/edge_maps/test/rgbr"
    # gt_mat_dir = "result/gt_mat"
    # os.makedirs(gt_mat_dir, exist_ok=True)

    # for file in os.listdir(gt_dir):
    #     if file.endswith(".png") or file.endswith(".jpg"):
    #         img = cv2.imread(os.path.join(gt_dir, file), cv2.IMREAD_GRAYSCALE)
    #         img = img.astype(np.float32) / 255.0
    #         dummy_seg = np.zeros_like(img, dtype=np.uint8)
    #         save_path = os.path.join(gt_mat_dir, os.path.splitext(file)[0] + ".mat")
    #         savemat(save_path, {
    #             "groundTruth": {
    #                 "Boundaries": img,
    #                 "Segmentation": dummy_seg
    #             }
    #         })

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval.py <model>")
        sys.exit(1)
    model = sys.argv[1]
    generate_mat_files(model)
