import cv2
import numpy as np

def check_edge_percentage(image_path):
    """
    Calculate percentage of white to black pixels in an image using threshold of 127.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        float: percentage of white pixels to black pixels
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Apply threshold
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Count white and black pixels
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    
    # Calculate percentage of edge pixels (white) relative to total pixels
    percentage = (white_pixels / (white_pixels + black_pixels)) * 100 if (white_pixels + black_pixels) > 0 else 0
    
    # print(f"White pixels: {white_pixels}")
    # print(f"Black pixels: {black_pixels}") 
    # print(f"percentage (white:black): {percentage:.3f}")
    
    return percentage

def analyze_folder(folder_path):
    """
    Analyze all images in a folder and calculate edge percentage statistics.
    
    Args:
        folder_path (str): Path to folder containing images
        
    Returns:
        dict: Dictionary containing percentage statistics
    """
    import os
    from statistics import mean, median, stdev
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
        
    percentages = []
    
    print("\nAnalyzing images...")
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            percentage = check_edge_percentage(img_path)
            percentages.append(percentage)
            # print(f"\nImage: {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
            
    if not percentages:
        raise ValueError("No valid percentages calculated")
        
    # Calculate statistics
    stats = {
        'mean': mean(percentages),
        'median': median(percentages),
        'min': min(percentages),
        'max': max(percentages),
        'std': stdev(percentages)
    }
    
    print("\nSummary Statistics:")
    print(f"Mean percentage: {stats['mean']:.3f}")
    print(f"Median percentage: {stats['median']:.3f}") 
    print(f"Min percentage: {stats['min']:.3f}")
    print(f"Max percentage: {stats['max']:.3f}")
    print(f"Standard deviation: {stats['std']:.3f}")
    
    return stats

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_image_edges.py <folder_path>")
        sys.exit(1)
    analyze_folder(sys.argv[1])
