import os
import cv2
import numpy as np
from scipy.io import savemat

# === Paths ===
image_dir = "result/BIPED2UDED/5_model_original"
mat_dir = "result/result_mat"
os.makedirs(mat_dir, exist_ok=True)
for file in os.listdir(image_dir):
    if file.endswith(".png") or file.endswith(".jpg"):
        img = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0  # Normalize if needed
        save_path = os.path.join(mat_dir, os.path.splitext(file)[0] + ".mat")
        savemat(save_path, {"result": img})

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
        # Wrap the dictionary in a 1x1 numpy array with dtype=object
        # gt = np.array([[{'Boundaries': img, 'Segmentation': dummy_seg}]], dtype=object)
        # savemat(save_path, {"groundTruth": gt})