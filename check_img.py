import cv2
import numpy as np
from scipy.io import loadmat

# File paths (adjust as needed)
result_mat_path = "result/result_mat/RGB_214.mat"
gt_mat_path = "result/gt_mat/RGB_214.mat"

# Load MAT files
result_data = loadmat(result_mat_path)
gt_data = loadmat(gt_mat_path)

# Extract the edge map from the result file (saved with key "result")
result_img = result_data["result"]

# Extract the ground truth edge map.
# If you saved it as an array of dicts, for example with a field "Boundaries":
gt_struct = gt_data["groundTruth"][0, 0]
gt_img = gt_struct['Boundaries']

print("Keys:", list(gt_data.keys()))

# Assuming groundTruth is the key for the structure
gt = gt_data["groundTruth"]
print("Shape of groundTruth:", gt.shape)  # Expected shape: (1, N) or similar

# Inspect the first element (assuming it's stored as a 1x1 structure)
first_gt = gt[0, 0]
print("Fields in the first ground truth structure:", first_gt.dtype.names)

# Convert images from float (assumed [0,1]) to 8-bit scale [0,255]
result_uint8 = (result_img * 255).astype(np.uint8)
gt_uint8 = (gt_img * 255).astype(np.uint8)

# # Display the images
# cv2.imshow("Result", result_uint8)
# cv2.imshow("Ground Truth", gt_uint8)
# cv2.waitKey(0)
# cv2.destroyAllWindows()