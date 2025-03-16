import os
import cv2
import numpy as np
import random
# Original synthetic dataset module

class DrawPng:
    def __init__(self, width, height, png_dir):
        self.width = width
        self.height = height
        self.png_dir = png_dir

    def __call__(self, num_objects):
        edge_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        fill_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        overlap_threshold = 0.3  # Maximum overlap ratio allowed for a region

        for _ in range(num_objects):
            """Pick a random PNG from png_dir, threshold it, and place it on fill_mask at a random location."""
            png_files = [f for f in os.listdir(self.png_dir) if f.endswith('.png')]
            if not png_files:
                print("No PNGs available")
                return

            chosen_png = random.choice(png_files)
            png_path = os.path.join(self.png_dir, chosen_png)

            # Read and process the PNG
            img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            scale = random.uniform(0.15, 0.5)
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            angle = random.uniform(0, 360)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

            H, W = fill_mask.shape
            max_y = H - new_h
            max_x = W - new_w
            if max_y < 0 or max_x < 0:
                continue  # Skip if the PNG doesn't fit

            for _ in range(100):  # Attempt to find a suitable placement
                top_y = random.randint(0, max_y)
                left_x = random.randint(0, max_x)

                region = fill_mask[top_y:top_y+new_h, left_x:left_x+new_w]
                overlap_ratio = np.sum((region > 0) & (mask > 0)) / np.sum(mask > 0)

                if overlap_ratio <= overlap_threshold:
                    # Place the mask if overlap is within the threshold
                    fill_mask[top_y:top_y+new_h, left_x:left_x+new_w] = np.where(mask > 0, 255, region)
                    break
            else:
                print("Could not place object within constraints after 100 attempts.")

        return edge_mask, fill_mask, []


# import os
# from PIL import Image
# import numpy as np
# from scipy.ndimage import binary_erosion

# def process_image(image_path):
#     # Open the image and convert it to grayscale
#     img = Image.open(image_path).convert('L')
#     # Convert the grayscale image to a binary image (0 for black, 255 for white)
#     binary_img = np.array(img) < 128  # Thresholding at 128

#     # Perform erosion
#     eroded_img = binary_erosion(binary_img, structure=np.ones((3, 3)))

#     # Compute the difference: pixels that are lit up in binary_img but not in eroded_img
#     edge_img = binary_img & ~eroded_img

#     # Convert back to the original format
#     output_img = Image.fromarray((edge_img * 255).astype(np.uint8))

#     # Save the eroded image
#     output_filename = f"./png/gt/eroded_{os.path.basename(image_path)}"
#     output_img.save(output_filename)
#     print(f"Processed and saved: {output_filename}")

# def main():
#     current_directory = os.getcwd()
#     current_directory =  "./png/input/"
#     for filename in os.listdir(current_directory):
#         if filename.endswith('.png'):
#             print("Found: ", filename)
#             process_image("./png/input/"+filename)

# if __name__ == "__main__":
#     main()