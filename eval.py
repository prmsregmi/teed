import os
import cv2
import sys
import numpy as np
import toml
from teed import main
from scipy.io import savemat
import subprocess
import concurrent.futures

# Global variables
original_config = toml.load("config.toml")
def call_ods_ois(model, mat_dir):
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

def generate_mat_files(results_path):
    # === Paths ===
    image_dir = results_path
    mat_dir = os.path.join(results_path, "result_mat")
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


def run_inference():
    global original_config
    try:
        for i in range(0, original_config["training"]["epochs"]):
            config = toml.load("config.toml")
            config["general"]["is_testing"] = True
            config["training"]["checkpoint_data"] = f"{i}_model.pth"
            config["paths"]["output_dir"] = f"checkpoints/CLASSIC/{i}/"
            with open("config.toml", "w") as f:
                toml.dump(config, f)
            main()
    finally:
        # Restore original config
        with open("config.toml", "w") as f:
            toml.dump(original_config, f)

def run_eval():
    global original_config
    epochs = original_config["training"]["epochs"]

    def process_epoch(i):
        img_dir = f"result/checkpoints/CLASSIC/{i}/{i}_model.pth/fused/"
        generate_mat_files(img_dir)
        call_ods_ois("Classic",  os.path.join(results_path, "result_mat"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=epochs) as executor:
        futures = [executor.submit(process_epoch, i) for i in range(2, 3)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <model> <checkpoint_path>")
        sys.exit(1)
    model = sys.argv[1]
    results_path = sys.argv[2]
    generate_mat_files(results_path)
    call_ods_ois(model, os.path.join(results_path, "result_mat"))


