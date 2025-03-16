import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from .draw_shapes import DrawShape
from .draw_png import DrawPng

# Original synthetic dataset module
def load_and_smooth_image(image_path):
    """Load an image and apply extreme smoothing to remove edges"""
    img = cv2.imread(str(image_path))
    kernel_size = 31
    for _ in range(3):
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img


def get_random_color(background_color):
    """Output a random scalar with contrast"""
    color = random.randint(0, 255)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color


def generate_background(size, nb_blobs=100, min_rad_ratio=0.01,
                        max_rad_ratio=0.05, min_kernel_size=100, max_kernel_size=300):
    """Generate background as in the original code"""
    img = np.zeros(size, dtype=np.uint8)
    dim = max(size)

    # Create random binary noise
    cv2.randu(img, 0, 255)
    cv2.threshold(img, random.randint(0, 256), 255, cv2.THRESH_BINARY, img)

    # Get background color and add blobs
    background_color = int(np.mean(img))
    for _ in range(nb_blobs):
        x = random.randint(0, size[1])
        y = random.randint(0, size[0])
        radius = random.randint(int(dim * min_rad_ratio), int(dim * max_rad_ratio))
        color = get_random_color(background_color)
        cv2.circle(img, (x, y), radius, color, -1)

    # Apply final blur
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    cv2.blur(img, (kernel_size, kernel_size), img)
    return img


def get_outer_edge_mask(fill_mask):
    """
    Returns a binary mask of only the outer edges of the shapes present in fill_mask.
    Ignores any overlapping boundaries within shapes.
    """
    # Ensure fill_mask is binary [0 or 255]
    # (In case your fill_mask is already binary, you can skip this threshold.)
    _, fill_mask_bin = cv2.threshold(fill_mask, 127, 255, cv2.THRESH_BINARY)

    # Create a structuring element (3x3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Erode the fill_mask to shrink shapes
    eroded_mask = cv2.erode(fill_mask_bin, kernel, iterations=1)

    # Subtract eroded_mask from fill_mask_bin to isolate outer boundaries
    outer_edge_mask = cv2.subtract(fill_mask_bin, eroded_mask)

    # Dilation to thicken the outer edge
    # Using the same kernel or a slightly larger one (e.g., 3x3 or 5x5)
    # outer_edge_mask = cv2.dilate(outer_edge_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    return outer_edge_mask


def generate_synthetic_sample(image_size, num_synthetic, preprocessing_size, output_dir, img_type, png_dir, texture_dir):
    """Generate synthetic sample with separate edge and fill masks"""
    height, width = image_size
    draw = DrawPng(width, height, png_dir) if img_type == 'png' else DrawShape(width, height, ['line', 'ellipse', 'polygon', 'star', 'cube'])
    
    print("Generating synthetic samples...")
    texture_dir = Path(texture_dir)
    texture_files = list(texture_dir.glob('*'))  # list all texture files
    for idx in tqdm(range(num_synthetic)):
        # Generate background
        background = generate_background((height, width))
        background = np.stack([background, background, background], axis=2)

        edge_mask, fill_mask, points = draw(random.randint(3, 7))

        # Convert fill mask to 3D and normalize
        fill_mask_3d = np.stack([fill_mask, fill_mask, fill_mask], axis=2) / 255.0

        if random.randint(1, 10) <= 6: #70% texture foreground
            # Use a texture
            texture_file = random.choice(texture_files)
            texture = cv2.imread(str(texture_file))
            texture = cv2.resize(texture, (width, height))
            fill_pattern = texture.astype(np.float32)
        else:
            # Use a random color
            background_color = int(np.mean(background))
            fill_color = get_random_color(background_color)
            fill_pattern = np.array([fill_color, fill_color, fill_color], dtype=np.float32)
        
            # And use a texture for the background instead of the generated one
            if random.randint(1, 10) <= 6:
                bg_texture = cv2.imread(str(random.choice(texture_files)))
                background = cv2.resize(bg_texture, (width, height))

        
        # Blend the background with the fill (texture or color) using the mask
        image = background.astype(np.float32) * (1 - fill_mask_3d) + fill_pattern * fill_mask_3d
        image = image.astype(np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # # Create high contrast between filled shapes and background
        # background_color = int(np.mean(background))
        # fill_color = get_random_color(background_color)
        # fill_color_3d = np.array([fill_color, fill_color, fill_color])

        # # Blend background with filled shapes
        # image = background * (1 - fill_mask_3d) + fill_color_3d * fill_mask_3d

        # # Add final blur to background but not to edges
        # image = image.astype(np.uint8)
        # image = cv2.GaussianBlur(image, (5, 5), 0)

        outer_edges = get_outer_edge_mask(fill_mask)

        # Resize if needed
        if preprocessing_size != image_size:
            image = cv2.resize(image, preprocessing_size[::-1])
            outer_edges = cv2.resize(outer_edges, preprocessing_size[::-1], interpolation=cv2.INTER_CUBIC)

        base_name = f'synthetic_shapes_{idx:04d}'
        cv2.imwrite(str(output_dir / 'images' / f'{base_name}.png'), image)
        cv2.imwrite(str(output_dir / 'masks' / f'{base_name}.png'), outer_edges)


def generate_real_image_sample(image_size, num_real, real_img_dir, preprocessing_size, output_dir, img_type, png_dir):
    """Generate real image based sample with separate edge and fill masks"""
    height, width = image_size
    draw = DrawPng(width, height, png_dir) if img_type == 'png' else DrawShape(width, height, ['line', 'ellipse', 'polygon', 'star', 'cube'])
    
    real_img_dir = Path(real_img_dir)
    image_files = list(real_img_dir.glob('*.jpg')) + list(real_img_dir.glob('*.png'))

      # Generate real image based samples
    print("Generating real image based samples...")
    for idx in tqdm(range(num_real)):
        img_a_path, img_b_path = random.sample(image_files, 2)
        img_a = load_and_smooth_image(img_a_path)
        img_b = load_and_smooth_image(img_b_path)
        
        # Resize images if needed
        if img_a.shape[:2] != (height, width):
            img_a = cv2.resize(img_a, (width, height))

        if img_b.shape[:2] != (height, width):
            img_b = cv2.resize(img_b, (width, height))

        # Create masks
        edge_mask = np.zeros((height, width), dtype=np.uint8)
        fill_mask = np.zeros((height, width), dtype=np.uint8)

        edge_mask, fill_mask, points = draw(random.randint(2, 5))

        # Create final image by blending
        fill_mask_3d = np.stack([fill_mask, fill_mask, fill_mask], axis=2) / 255.0
        image = img_a * fill_mask_3d + img_b * (1 - fill_mask_3d)

        outer_edges = get_outer_edge_mask(fill_mask)

        # Resize if needed
        if preprocessing_size != image_size:
            image = cv2.resize(image, preprocessing_size[::-1])
            outer_edges = cv2.resize(outer_edges, preprocessing_size[::-1], interpolation=cv2.INTER_CUBIC)

        base_name = f'real_shapes_{idx:04d}'
        cv2.imwrite(str(output_dir / 'images' / f'{base_name}.png'), image)
        cv2.imwrite(str(output_dir / 'masks' / f'{base_name}.png'), outer_edges)


def process_and_save_images(real_img_dir, png_dir, output_dir, img_type, num_synthetic, num_real,
                            texture_dir, image_size=(720, 1280), preprocessing_size=(720, 1280)):
    """Generate and save both synthetic and real-image-based samples"""
    output_dir = Path(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)

        
    generate_real_image_sample(image_size, num_real, real_img_dir, preprocessing_size, output_dir, img_type, png_dir)
    generate_synthetic_sample(image_size, num_synthetic, preprocessing_size, output_dir, img_type, png_dir, texture_dir)

def generate_synthetic():
    img_type = "png"  # shape, png

    # Example usage
    real_img_dir = "./data/BIPED/BIPED/edges/imgs/train/rgbr/real"
    png_dir = "./data/png/input"
    output_dir = "./data/synthetic_train"
    texture_dir = "./data/textures"

    process_and_save_images(
            real_img_dir=real_img_dir,
            png_dir=png_dir,
            output_dir=output_dir,
            img_type=img_type,  # Use the terminal argument or default
            num_synthetic=100,  # Number of synthetic background samples
            num_real=100,  # Number of real image-based samples
            texture_dir=texture_dir,
            # image_size=(1280, 1280),
            # preprocessing_size=(512, 512)
    )

if __name__ == "__main__":
    generate_synthetic()