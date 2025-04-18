import os
import cv2
import sys
import numpy as np
import toml
from scipy.io import savemat
import subprocess
import concurrent.futures

def generate_mat_files(image_dir):
    mat_dir = os.path.join(image_dir, "result_mat")
    os.makedirs(mat_dir, exist_ok=True)
    count = 0
    for file in os.listdir(image_dir):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            img = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_GRAYSCALE) 
            img = img.astype(np.float32) / 255.0  # Normalize if needed
            save_path = os.path.join(mat_dir, os.path.splitext(file)[0] + ".mat")
            savemat(save_path, {"result": img})
            count += 1
    print(f"{count} files saved as mat.")
    return mat_dir

def call_ods_ois(model, results_path):
    mat_dir = generate_mat_files(results_path)
    # Path to the external Python interpreter if different from the main project's
    external_python = ".venv/bin/python3"
    
    # Build the command
    cmd = [
        external_python,
        "main.py",
        "--alg", "TEED " + model,
        "--model_name_list", model,
        "--result_dir", "../../" + mat_dir,
        "--save_dir", "../../" + mat_dir + "/eval_output",
        "--gt_dir", "../../data/UDED/gt_mat",
        "--key", "result",
        "--file_format", ".mat",
        "--workers", "-1"
    ]
    
    # Call the script, changing to its directory
    subprocess.run(cmd, cwd="external/edge_eval")


    ## Code for converting GT to mat

    # gt_dir = "data/xray/gt/"
    # gt_mat_dir = "data/xray/gt_mat"
    # os.makedirs(gt_mat_dir, exist_ok=True)

    # for filename in os.listdir(gt_dir):
    #     if filename.lower().endswith(('.png', '.jpg')):
    #         img = cv2.imread(os.path.join(gt_dir, filename), cv2.IMREAD_GRAYSCALE)
    #         if img is None:
    #             continue
    #         img_uint8 = img.astype(np.uint8)
    #         # Create a (1,1) object array holding a dictionary with key "Boundaries"
    #         groundTruth = np.array([[{'Boundaries': img_uint8}]], dtype=object)
    #         save_path = os.path.join(gt_mat_dir, os.path.splitext(filename)[0] + ".mat")
    #         savemat(save_path, {'groundTruth': groundTruth})

def run_eval_batch(folder_name,multithread=False):
    original_config = toml.load("config.toml")
    epochs = original_config["training"]["epochs"]

    def process_epoch(i):
        img_dir = f"result/{folder_name}/CLASSIC/{i}/{i}_model.pth/fused/"
        call_ods_ois("CLASSIC", img_dir)

    _loops = range(0, epochs)
    if multithread:
        with concurrent.futures.ThreadPoolExecutor(max_workers=epochs) as executor:
            futures = [executor.submit(process_epoch, i) for i in _loops]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for i in _loops:
            process_epoch(i)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <model> <checkpoint_path>")
        sys.exit(1)
    model = sys.argv[1]
    results_path = sys.argv[2]
    call_ods_ois(model, os.path.join(results_path))
