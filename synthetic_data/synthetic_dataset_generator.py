import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from .draw_shapes import DrawShape
from .draw_png import DrawPng
import argparse
import scipy.interpolate
from perlin_noise import PerlinNoise

# Original synthetic dataset module
def load_and_smooth_image(image_path):
    """Load an image and apply extreme smoothing to remove edges"""
    img = cv2.imread(str(image_path))
    kernel_size = 31
    for _ in range(3):
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img

def load_and_smooth_image_bgr(image_path):
    """Load an image and apply moderate smoothing."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        # Return a default gray image if loading fails
        return np.full((100, 100, 3), 128, dtype=np.uint8)
    # Reduced smoothing compared to original example
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def get_random_color_bgr(background_avg_bgr):
    """Output a random BGR color tuple with contrast"""
    color_bgr = [random.randint(0, 255) for _ in range(3)]

    # Calculate simple luminance difference for contrast check
    bg_lum = 0.299 * background_avg_bgr[2] + 0.587 * background_avg_bgr[1] + 0.114 * background_avg_bgr[0]
    fg_lum = 0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]

    if abs(fg_lum - bg_lum) < 40:  # Check contrast based on luminance
        # If contrast is too low, invert the color or shift significantly
        color_bgr = [(c + 128) % 256 for c in color_bgr]

    return tuple(color_bgr)

def generate_gradient_fill(width, height):
    """Generates a random linear gradient BGR image."""
    start_color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    end_color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)

    gradient = np.zeros((height, width, 3), dtype=np.float32)
    direction = random.random() # 0 for horizontal, >0.5 for vertical

    for i in range(height if direction > 0.5 else width):
        ratio = i / (height if direction > 0.5 else width)
        color = start_color * (1 - ratio) + end_color * ratio
        if direction > 0.5:
            gradient[i, :, :] = color
        else:
            gradient[:, i, :] = color

    return gradient.astype(np.uint8)

def generate_noise_background(size, channels=3):
    """Generates a simple Gaussian noise background."""
    height, width = size
    mean = random.uniform(50, 200)
    sigma = random.uniform(10, 50)
    noise = np.random.normal(mean, sigma, (height, width, channels))
    noise = np.clip(noise, 0, 255)
    return noise.astype(np.uint8)

def generate_perlin_background(size, octaves=4, seed=None):
    height, width = size
    if seed is None:
        seed = random.randint(0, 1000)
    noise = PerlinNoise(octaves=octaves, seed=seed)
    scale = random.uniform(0.01, 0.05) # Adjust scale for different patterns
    pic = [[noise([i*scale, j*scale]) for j in range(width)] for i in range(height)]
    pic_arr = np.array(pic)
    # Normalize to 0-255
    pic_norm = ((pic_arr - np.min(pic_arr)) / (np.max(pic_arr) - np.min(pic_arr))) * 255
    img = cv2.cvtColor(pic_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Apply random color tint
    tint = np.array(get_random_color_bgr((128,128,128)), dtype=float) / 255.0
    img = (img * tint).astype(np.uint8)
    return img

def get_random_color(background_color):
    """Output a random scalar with contrast"""
    color = random.randint(0, 255)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color

def generate_background(size, real_background_dir=None):
    """Generates a background: noise, gradient, or real image patch."""
    height, width = size
    choice = random.random()

    if choice < 0.4 and real_background_dir and real_background_dir.exists():
        try:
            bg_files = list(real_background_dir.glob('*.jpg')) + list(real_background_dir.glob('*.png'))
            if bg_files:
                bg_path = random.choice(bg_files)
                img = cv2.imread(str(bg_path))
                if img is not None:
                    # Simple resize, could also do random crop
                    return cv2.resize(img, (width, height))
                else:
                    print(f"Warning: Failed to load background {bg_path}")
        except Exception as e:
            print(f"Error loading background image: {e}")
        # Fallback if real image loading fails
        return generate_noise_background(size)

    elif choice < 0.7:
         return generate_gradient_fill(width, height)
    # elif choice < 0.8: # Add Perlin noise if desired and library installed
    #     return generate_perlin_background(size)
    else:
        return generate_noise_background(size)

def add_simple_shadow(background, fill_mask, offset=(5, 5), blur_kernel=(15, 15), strength=0.6):
    """Adds a simple drop shadow to the background."""
    shadow_mask = cv2.GaussianBlur(fill_mask, blur_kernel, 0)

    # Create offset shadow mask
    h, w = fill_mask.shape
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    offset_shadow_mask = cv2.warpAffine(shadow_mask, M, (w, h))

    # Apply shadow to background (darken)
    shadow_mask_3d = np.stack([offset_shadow_mask]*3, axis=2) / 255.0
    shadowed_bg = background.astype(np.float32) * (1 - shadow_mask_3d * strength)
    return np.clip(shadowed_bg, 0, 255).astype(np.uint8)

def apply_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """Applies random color jittering to the image."""
    img = image.astype(np.float32) / 255.0

    # Brightness
    if random.random() < 0.5:
        img += random.uniform(-brightness, brightness)

    # Contrast
    if random.random() < 0.5:
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * random.uniform(1 - contrast, 1 + contrast) + mean

    img = np.clip(img, 0, 1)

    # Saturation/Hue (Convert to HSV)
    if random.random() < 0.5 or random.random() < 0.5:
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        # Saturation
        if random.random() < 0.5:
            hsv[:, :, 1] *= random.uniform(1 - saturation, 1 + saturation)
        # Hue
        if random.random() < 0.5:
            hsv[:, :, 0] += random.uniform(-hue * 180, hue * 180)
            hsv[:, :, 0] %= 180 # Hue wraps around

        hsv = np.clip(hsv, 0, [179, 255, 255]) # Valid HSV ranges
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    return np.clip(img * 255, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, mean=0, sigma_range=(5, 20)):
    """Adds Gaussian noise to the image."""
    sigma = random.uniform(*sigma_range)
    h, w, c = image.shape
    noise = np.random.normal(mean, sigma, (h, w, c))
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_random_blur(image, kernel_range=(3, 7)):
    """Applies Gaussian blur with a random kernel size."""
    if random.random() < 0.5: # Only apply blur sometimes
        k_size = random.randrange(kernel_range[0], kernel_range[1] + 1, 2) # Odd kernel size
        return cv2.GaussianBlur(image, (k_size, k_size), 0)
    return image


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


def get_enhanced_edge_mask(fill_mask, thickness_range=(1, 3), intensity_range=(0.7, 1.0),
                           blur_kernel_range=(1, 3)):
    """Returns a grayscale mask with variable thickness and intensity edges"""
    # Ensure fill_mask is binary [0 or 255]
    _, fill_mask_bin = cv2.threshold(fill_mask, 127, 255, cv2.THRESH_BINARY)

    # Randomize thickness parameters
    thickness = random.randint(thickness_range[0], thickness_range[1])
    blur_kernel = random.randint(blur_kernel_range[0], blur_kernel_range[1]) * 2 + 1

    # Create a structuring element with variable size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))

    # Erode the fill_mask to shrink shapes
    eroded_mask = cv2.erode(fill_mask_bin, kernel, iterations=1)

    # Subtract eroded_mask from fill_mask_bin to isolate boundaries
    outer_edge_mask = cv2.subtract(fill_mask_bin, eroded_mask)

    # Add variable blur to simulate realistic edge transitions
    outer_edge_mask = cv2.GaussianBlur(outer_edge_mask, (blur_kernel, blur_kernel), 0)

    # Randomize intensity
    intensity = random.uniform(intensity_range[0], intensity_range[1])
    outer_edge_mask = (outer_edge_mask * intensity).astype(np.uint8)

    return outer_edge_mask


def apply_edge_texturing(edge_mask, intensity_range=(0.05, 0.2)):
    """Apply realistic texture to edges"""
    # Convert to float for calculations
    edge_float = edge_mask.astype(np.float32) / 255.0

    # Only apply noise to edge pixels
    edge_pixels = edge_float > 0

    # Create noise pattern
    noise_intensity = random.uniform(intensity_range[0], intensity_range[1])
    noise = np.random.normal(0, noise_intensity, edge_mask.shape)

    # Apply noise only to edge areas
    textured_edge = edge_float.copy()
    textured_edge[edge_pixels] = np.clip(textured_edge[edge_pixels] + noise[edge_pixels], 0, 1)

    # Convert back to uint8
    return (textured_edge * 255).astype(np.uint8)


def generate_curved_shape(width, height, num_control_points=8, variability=0.4):
    """Generate a curved shape using spline interpolation"""
    # Generate random control points along a circle
    angles = np.linspace(0, 2 * np.pi, num_control_points, endpoint=False)
    center_x, center_y = width // 2, height // 2

    # Random radius with some variation
    base_radius = min(width, height) // 4

    # Generate control points with varying radius
    control_points = []
    for angle in angles:
        # Add randomness to radius
        radius = base_radius * (1 + random.uniform(-variability, variability))
        x = center_x + int(radius * np.cos(angle))
        y = center_y + int(radius * np.sin(angle))
        control_points.append((x, y))

    # Add the first point again to close the curve
    control_points.append(control_points[0])

    # Convert to numpy array for spline interpolation
    points = np.array(control_points, dtype=np.int32)

    # Generate a smooth curve with many points
    x, y = points[:, 0], points[:, 1]
    tck, u = scipy.interpolate.splprep([x, y], s=0, per=True)

    # Create more interpolated points
    u_new = np.linspace(0, 1, 100)
    x_new, y_new = scipy.interpolate.splev(u_new, tck)

    # Create a mask and draw the curve
    mask = np.zeros((height, width), dtype=np.uint8)
    curve_points = np.column_stack([x_new, y_new]).astype(np.int32)
    curve_points = curve_points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [curve_points], 255)

    return mask


def create_edge_hierarchy(image, edge_mask):
    """Create hierarchy of edges based on image content"""
    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Canny edge detection with conservative thresholds
    canny_edges = cv2.Canny(gray, 50, 150)

    # Dilate Canny edges to ensure overlap with our edge mask
    kernel = np.ones((3, 3), np.uint8)
    dilated_canny = cv2.dilate(canny_edges, kernel, iterations=1)

    # Divide edges into primary and secondary
    # Primary: where our edge mask overlaps with Canny edges (strong edges)
    primary_edges = cv2.bitwise_and(edge_mask, dilated_canny)

    # Secondary: edges in our mask that aren't primary
    secondary_edges = cv2.subtract(edge_mask, primary_edges)

    # Reduce intensity of secondary edges
    secondary_edges = (secondary_edges.astype(np.float32) * 0.5).astype(np.uint8)

    # Combine to create hierarchical edge map
    return cv2.add(primary_edges, secondary_edges)


def get_adaptive_edge_mask(fill_mask, image=None):
    """Generate edges with thickness based on shape complexity"""
    # Ensure binary fill mask
    _, fill_mask_bin = cv2.threshold(fill_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours to analyze shape complexity
    contours, _ = cv2.findContours(fill_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create empty edge mask
    edge_mask = np.zeros_like(fill_mask_bin)

    for contour in contours:
        # Calculate shape complexity (perimeter²/area)
        area = cv2.contourArea(contour)
        if area < 10:  # Skip very small contours
            continue

        perimeter = cv2.arcLength(contour, True)
        complexity = perimeter ** 2 / (4 * np.pi * area) if area > 0 else 1

        # Adjust thickness based on complexity (more complex = thinner)
        thickness = max(1, int(4 / np.sqrt(complexity)))

        # Draw contour with adaptive thickness
        cv2.drawContours(edge_mask, [contour], 0, 255, thickness)

    # Subtract filled area to get only the edge
    edge_only = cv2.subtract(edge_mask, fill_mask_bin)

    return edge_only


def apply_edge_variations(edge_mask, intensity_range=(0.7, 1.0)):
    """Apply realistic variations to edge intensity"""
    # Create noise pattern for variation
    h, w = edge_mask.shape
    noise = np.random.normal(0.5, 0.15, (h, w)).astype(np.float32)

    # Convert to float for calculations
    edge_float = edge_mask.astype(np.float32) / 255.0

    # Only apply to edge pixels
    edge_pixels = edge_float > 0

    # Random base intensity
    base_intensity = random.uniform(intensity_range[0], intensity_range[1])

    # Apply variations only to edge areas
    varied_edge = edge_float.copy()
    varied_edge[edge_pixels] = varied_edge[edge_pixels] * (base_intensity + 0.3 * (noise[edge_pixels] - 0.5))

    # Clip and convert back to uint8
    varied_edge = np.clip(varied_edge * 255, 0, 255).astype(np.uint8)

    return varied_edge


def add_edge_discontinuities(edge_mask, gap_probability=0.05, max_gap_length=3):
    """Add realistic gaps in edges based on image characteristics"""
    # Find edge pixels
    y_coords, x_coords = np.where(edge_mask > 0)
    edge_points = list(zip(y_coords, x_coords))

    # Create a copy of the edge mask
    result = edge_mask.copy()

    # Randomly select pixels to start gaps
    num_gaps = max(1, int(len(edge_points) * gap_probability))
    if len(edge_points) == 0:
        return result

    gap_starts = random.sample(edge_points, min(num_gaps, len(edge_points)))

    # Create gaps of random lengths
    for y, x in gap_starts:
        gap_length = random.randint(1, max_gap_length)

        # Find edge pixels in local neighborhood
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for i in range(gap_length):
                ny, nx = y + i * dy, x + i * dx
                if 0 <= ny < edge_mask.shape[0] and 0 <= nx < edge_mask.shape[1]:
                    result[ny, nx] = 0

    return result


def generate_context_aware_edges(image, fill_mask):
    """Generate edges that respond to the actual image content"""
    # Convert image to grayscale if it's color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Get basic edges using adaptive method
    basic_edges = get_adaptive_edge_mask(fill_mask)

    # Detect edges in the original image using Canny
    canny_edges = cv2.Canny(gray, 50, 150)
    dilated_canny = cv2.dilate(canny_edges, np.ones((3, 3), np.uint8))

    # Find where our edges overlap with real image edges (primary edges)
    primary_edges = cv2.bitwise_and(basic_edges, dilated_canny)

    # Edges that don't overlap are secondary
    secondary_edges = cv2.subtract(basic_edges, primary_edges)

    # Make primary edges stronger, secondary edges weaker
    primary_edges = cv2.multiply(primary_edges, 1.2)
    secondary_edges = cv2.multiply(secondary_edges, 0.6)

    # Combine for final hierarchical edge map
    combined_edges = cv2.add(primary_edges, secondary_edges)
    combined_edges = np.clip(combined_edges, 0, 255).astype(np.uint8)

    return combined_edges


def ensure_edge_presence(fill_mask, edge_mask, min_edge_ratio=0.1):
    """Ensure edges are present even for difficult shapes"""
    # Calculate perimeter of the shape (theoretical edge pixels)
    contours, _ = cv2.findContours(fill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected_edge_pixels = 0
    for contour in contours:
        expected_edge_pixels += cv2.arcLength(contour, True)

    # Count actual edge pixels
    actual_edge_pixels = np.count_nonzero(edge_mask)

    # If we have too few edge pixels compared to expected, enforce edges
    if actual_edge_pixels < expected_edge_pixels * min_edge_ratio:
        # Use morphological gradient as fallback
        kernel = np.ones((3, 3), np.uint8)
        morph_edge = cv2.morphologyEx(fill_mask, cv2.MORPH_GRADIENT, kernel)
        return morph_edge

    return edge_mask


def add_controlled_discontinuities(edge_mask, preserve_mask=None, gap_probability=0.05, max_gap_length=3):
    """Add discontinuities while preserving important areas marked in preserve_mask"""
    # Find edge pixels
    y_coords, x_coords = np.where(edge_mask > 0)
    edge_points = list(zip(y_coords, x_coords))

    # Create a copy of the edge mask
    result = edge_mask.copy()

    # Randomly select pixels to start gaps
    num_gaps = max(1, int(len(edge_points) * gap_probability))
    if len(edge_points) == 0:
        return result

    gap_starts = random.sample(edge_points, min(num_gaps, len(edge_points)))

    # Create gaps of random lengths
    for y, x in gap_starts:
        # Skip if this point should be preserved
        if preserve_mask is not None and preserve_mask[y, x] > 0:
            continue

        # Random gap length
        gap_length = random.randint(1, max_gap_length)

        # Find neighbors to determine edge direction
        neighbors = []
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < edge_mask.shape[0] and 0 <= nx < edge_mask.shape[1] and edge_mask[ny, nx] > 0:
                neighbors.append((ny, nx, dy, dx))

        if not neighbors:
            continue

        # Choose a random direction
        ny, nx, dy, dx = random.choice(neighbors)

        # Create the gap along this direction
        for i in range(gap_length):
            cy, cx = y + i * dy, x + i * dx
            if (0 <= cy < result.shape[0] and 0 <= cx < result.shape[1] and
                    result[cy, cx] > 0):
                # Don't create gaps in preserved areas
                if preserve_mask is not None and preserve_mask[cy, cx] > 0:
                    break
                result[cy, cx] = 0

    return result


def hybrid_edge_detection(fill_mask, image, add_discontinuities=False):
    """
    Hybrid edge detection using both geometric mask and final image content.
    This approach implicitly handles internal overlap edges if they create
    sufficient contrast in the final 'image'.

    Args:
        fill_mask (np.array): The combined binary mask (uint8) of the union
                              of all drawn shapes.
        image (np.array):     The final composited BGR image (uint8) after all
                              shapes, textures, colors, and effects are applied.
        add_discontinuities (bool): Whether to add random gaps to the edges.

    Returns:
        np.array: The final generated grayscale edge mask (uint8).
    """
    # 1. Get geometric edges from the combined fill mask
    #    - Ground truth silhouette (safety net)
    ground_truth_edges = get_outer_edge_mask(fill_mask)
    #    - Edges with adaptive thickness based on silhouette complexity
    #      (Pass image=None here, or the final image, depending on desired behavior.
    #      Using None focuses purely on geometry for this step)
    adaptive_geom_edges = get_adaptive_edge_mask(fill_mask, image=None) # Focus on geometry

    # 3. Apply intensity variations for realism to the adaptive geometric edges
    adaptive_geom_edges = apply_edge_variations(adaptive_geom_edges)

    # 4. Combine geometric edge masks (prioritize stronger signals)
    #    Start with the basic outer edge
    combined_geom_edges = ground_truth_edges.copy()
    #    Where adaptive edges are stronger, use their intensity
    mask = (adaptive_geom_edges > combined_geom_edges) & (adaptive_geom_edges > 0)
    combined_geom_edges[mask] = adaptive_geom_edges[mask]

    # 5. *** Refine edges using the FINAL IMAGE content ***
    #    This is where internal overlap edges can be detected if contrasty.
    final_refined_edges = combined_geom_edges.copy() # Start with geometry base
    if image is not None:
        # Convert final image to grayscale for Canny
        # Inside hybrid_edge_detection, Step 5
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ---> ADD BLUR HERE <---
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Small 3x3 blur

        # Use the blurred image for Canny
        # canny_edges = cv2.Canny(gray_blurred, 50, 150)  # Keep original thresholds OR tune them
        # OR Tune Thresholds: Try higher values to ignore weaker edges (more likely noise)
        canny_edges = cv2.Canny(gray_blurred, 100, 200)

        # Get Canny edges from the *actual pixel data* of the final image
        # These Canny edges will include outer edges AND internal edges
        # if the contrast between overlapping shapes/background is high enough.
        dilated_canny = cv2.dilate(canny_edges, np.ones((3, 3), np.uint8), iterations=1)

        # --- Hierarchy Creation ---
        # Primary edges: Where geometric proposal overlaps with image Canny edges
        # (Edges supported by both geometry and image content)
        primary_edges = cv2.bitwise_and(combined_geom_edges, dilated_canny)

        # Secondary edges: Geometric proposals not directly confirmed by Canny
        # (Edges based mainly on the shape masks)
        secondary_edges_geom = cv2.subtract(combined_geom_edges, primary_edges)

        # Tertiary edges (Optional but good): Canny edges not in geometric proposal
        # (Edges purely from image content - e.g., texture edges, internal overlaps missed by geom)
        tertiary_edges_canny = cv2.subtract(dilated_canny, combined_geom_edges)

        # --- Combine with weighting ---
        # Enhance primary edges (high confidence)
        primary_contribution = cv2.multiply(primary_edges.astype(np.float32), 1.2) # Make stronger
        # Keep secondary edges (medium confidence)
        secondary_contribution = cv2.multiply(secondary_edges_geom.astype(np.float32), 0.8) # Slightly reduce? Or keep 1.0?
        # Add tertiary edges (lower confidence, only from image)
        tertiary_contribution = cv2.multiply(tertiary_edges_canny.astype(np.float32), 0.3) # Weaker intensity

        # Combine all contributions
        final_refined_edges_float = primary_contribution + secondary_contribution + tertiary_contribution
        final_refined_edges = np.clip(final_refined_edges_float, 0, 255).astype(np.uint8)

        # Fallback: Ensure some edges exist if refinement wiped them out
        if np.count_nonzero(final_refined_edges) < 0.1 * np.count_nonzero(combined_geom_edges):
             final_refined_edges = combined_geom_edges # Revert if too much was lost


    # 6. Optionally add discontinuities (controlled gaps)
    if add_discontinuities:
        # Find corners from the original geometric edges to preserve them
        corners = cv2.goodFeaturesToTrack(ground_truth_edges, 50, 0.01, 10)
        corner_mask = np.zeros_like(ground_truth_edges)
        if corners is not None:
            corners = np.int32(corners)
            for corner in corners:
                x, y = corner.ravel()
                # Make preserved area slightly larger
                cv2.circle(corner_mask, (x, y), 7, 255, -1) # Increased radius

        # Add gaps to the refined edges, avoiding corner regions
        final_refined_edges = add_controlled_discontinuities(final_refined_edges,
                                                             preserve_mask=corner_mask,
                                                             gap_probability=0.08, # Adjust as needed
                                                             max_gap_length=4)    # Adjust as needed

    return final_refined_edges


def generate_improved_synthetic_sample(image_size, num_synthetic, preprocessing_size, output_dir,
                                       img_type, png_dir, texture_dir, real_background_dir=None):
    """Generate synthetic samples with improved realism."""
    height, width = image_size
    # Initialize Draw class based on img_type
    draw = DrawPng(width, height, png_dir) if img_type == 'png' else DrawShape(width, height,
                                                                               ['line', 'ellipse', 'polygon', 'star', 'cube'])

    print("Generating improved synthetic samples...")
    texture_dir = Path(texture_dir)
    texture_files = list(texture_dir.glob('*.*')) # Allow different texture extensions
    real_background_dir = Path(real_background_dir) if real_background_dir else None

    for idx in tqdm(range(num_synthetic)):
        # 1. Generate Background (more variety)
        background = generate_background((height, width), real_background_dir)
        background = cv2.resize(background, (width, height)) # Ensure size consistency

        # Calculate average background color for contrast checks
        background_avg_bgr = np.mean(background, axis=(0, 1))

        # 1b. Optionally add shadow placeholder before shapes
        # Create a dummy mask for shadow calculation if needed early,
        # or calculate shadow after shapes are drawn. We'll add it later here.
        background_with_shadow = background.copy() # Start with original bg


        # 2. Generate Shapes (potentially multiple)
        num_shapes = random.randint(1, 3) # Draw 1 to 3 shapes
        final_fill_mask = np.zeros((height, width), dtype=np.uint8)
        final_edge_mask = np.zeros((height, width), dtype=np.uint8) # For basic edge accumulation if needed
        composite_image = background.astype(np.float32)

        # --- Shadow preparation ---
        all_shapes_fill_mask = np.zeros((height, width), dtype=np.uint8)


        drawn_shape_patterns = [] # Store pattern for each shape

        for _ in range(num_shapes):
            # Generate single shape mask
            # draw() should return edge_mask, fill_mask, points
            try:
                # Increase complexity sometimes
                num_vertices_or_complexity = random.randint(3, 15 if isinstance(draw, DrawShape) else 7)
                edge_mask_i, fill_mask_i, _ = draw(num_vertices_or_complexity)
            except Exception as e:
                 print(f"Error during drawing shape: {e}")
                 continue # Skip this shape if drawing fails

            # Prevent drawing over everything
            fill_mask_i = cv2.bitwise_and(fill_mask_i, cv2.bitwise_not(final_fill_mask))
            if np.sum(fill_mask_i) < 50: # Skip tiny or fully occluded shapes
                 continue

            final_fill_mask = cv2.bitwise_or(final_fill_mask, fill_mask_i)
            # Accumulate basic edges if needed, though hybrid_edge_detection is preferred
            final_edge_mask = cv2.bitwise_or(final_edge_mask, edge_mask_i)
            all_shapes_fill_mask = cv2.bitwise_or(all_shapes_fill_mask, fill_mask_i) # For shadow


            # 3. Apply Texture, Color, or Gradient to the CURRENT shape
            fill_pattern = None
            pattern_choice = random.random()

            if pattern_choice < 0.6 and texture_files: # 60% texture
                try:
                    texture_file = random.choice(texture_files)
                    texture = cv2.imread(str(texture_file))
                    if texture is not None:
                        texture = cv2.resize(texture, (width, height))
                        fill_pattern = texture.astype(np.float32)
                    else:
                        print(f"Warning: Failed to load texture {texture_file}")
                except Exception as e:
                    print(f"Error loading texture: {e}")
                if fill_pattern is None: # Fallback if texture fails
                    pattern_choice = 0.7 # Force gradient or color

            if fill_pattern is None: # Gradient or Solid Color if no texture or failed
                if pattern_choice < 0.85: # 25% Gradient
                     gradient_img = generate_gradient_fill(width, height)
                     fill_pattern = gradient_img.astype(np.float32)
                else: # 15% Solid Color
                     fill_color_bgr = get_random_color_bgr(background_avg_bgr)
                     # Create a BGR image with the fill color
                     fill_pattern = np.full((height, width, 3), fill_color_bgr, dtype=np.float32)

            drawn_shape_patterns.append({
                'mask': fill_mask_i,
                'pattern': fill_pattern
            })

        # 3b. Apply Simple Shadow based on all shapes
        if random.random() < 0.4: # Add shadow 40% of the time
             background_with_shadow = add_simple_shadow(background, all_shapes_fill_mask)

        # 4. Composite shapes onto background (potentially with shadow)
        # Start with the (potentially shadowed) background
        final_image = background_with_shadow.astype(np.float32)

        # Composite shapes layer by layer (order matters if they overlap, random order here)
        random.shuffle(drawn_shape_patterns)
        for item in drawn_shape_patterns:
             mask_i = item['mask']
             pattern_i = item['pattern']
             fill_mask_3d = np.stack([mask_i, mask_i, mask_i], axis=2) / 255.0
             final_image = final_image * (1 - fill_mask_3d) + pattern_i * fill_mask_3d

        # Ensure correct type before proceeding
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)

        # 5. Post-Processing Effects
        # Apply color jitter first
        if random.random() < 0.5: # 50% chance
             final_image = apply_color_jitter(final_image)

        # Apply blur
        final_image = apply_random_blur(final_image) # Blur happens fairly often

        # Apply noise last
        if random.random() < 0.3: # 30% chance
             final_image = add_gaussian_noise(final_image)


        # 6. Generate Edges using Hybrid Approach on the *final* image and *final* fill mask
        add_gaps = random.random() < 0.3
        # Use the combined fill mask of all shapes
        improved_edges = hybrid_edge_detection(final_fill_mask, final_image, add_discontinuities=add_gaps)

        # 6b. Edge Realism: Randomly blur edges slightly
        if random.random() < 0.25: # 25% chance
            blur_amount = random.choice([3, 5])
            improved_edges = cv2.GaussianBlur(improved_edges, (blur_amount, blur_amount), 0)
            # Optional: Re-normalize blurred edges if needed, e.g., enhance contrast
            # improved_edges = cv2.normalize(improved_edges, None, 0, 255, cv2.NORM_MINMAX)


        # # 7. Resize if needed (apply to both image and mask)
        # if preprocessing_size != image_size:
        #     final_image = cv2.resize(final_image, preprocessing_size[::-1], interpolation=cv2.INTER_LINEAR)
        #     # Use INTER_NEAREST or INTER_LINEAR for masks usually, CUBIC can introduce artifacts
        #     improved_edges = cv2.resize(improved_edges, preprocessing_size[::-1], interpolation=cv2.INTER_LINEAR)
        #     improved_edges = np.clip(improved_edges, 0, 255).astype(np.uint8) # Ensure valid range after resize


        # 8. Save the images
        base_name = f'synthetic_shapes_{idx:04d}'
        try:
            cv2.imwrite(str(output_dir / 'imgs/train/rgbr/real' / f'{base_name}.png'), final_image)
            cv2.imwrite(str(output_dir / 'edge_maps/train/rgbr/real' / f'{base_name}.png'), improved_edges)
        except Exception as e:
            print(f"Error saving image/mask {base_name}: {e}")


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
        cv2.imwrite(str(output_dir / 'imgs/train/rgbr/real' / f'{base_name}.png'), image)
        cv2.imwrite(str(output_dir / 'edge_maps/train/rgbr/real' / f'{base_name}.png'), outer_edges)


def process_and_save_images(real_img_dir, png_dir, output_dir, img_type, num_synthetic, num_real,
                            texture_dir, image_size=(720, 1280), preprocessing_size=(720, 1280)):
    """Generate and save both synthetic and real-image-based samples"""
    output_dir = Path(output_dir)
    (output_dir / 'imgs/train/rgbr/real').mkdir(parents=True, exist_ok=True)
    (output_dir / 'edge_maps/train/rgbr/real').mkdir(parents=True, exist_ok=True)

        
    generate_real_image_sample(image_size, num_real, real_img_dir, preprocessing_size, output_dir, img_type, png_dir)
    generate_improved_synthetic_sample(image_size, num_synthetic, preprocessing_size, output_dir, img_type, png_dir, texture_dir)

def generate_synthetic(real, synthetic):
    img_type = "png"  # shape, png

    # Example usage
    real_img_dir = "./data/BIPED/BIPED/edges/imgs/train/rgbr/real"
    png_dir = "./data/png/input"
    output_dir = "./data/synthetic_train/BIPED/edges"
    texture_dir = "./data/textures"

    process_and_save_images(
            real_img_dir=real_img_dir,
            png_dir=png_dir,
            output_dir=output_dir,
            img_type=img_type,  # Use the terminal argument or default
            num_synthetic=synthetic,  # Number of synthetic background samples
            num_real=real,  # Number of real image-based samples
            texture_dir=texture_dir,
            # image_size=(1280, 1280),
            # preprocessing_size=(512, 512)
    )

if __name__ == "__main__":
    generate_synthetic(250, 500)