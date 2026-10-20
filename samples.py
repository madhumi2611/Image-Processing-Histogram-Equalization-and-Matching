"""
samples.py - Provides built-in sample images for instant testing
Generates beautiful synthetic photographic scenes (low contrast, high contrast, gradients)
so users can experiment immediately even without uploading their own photos.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_sample_low_contrast() -> np.ndarray:
    """
    Generates a realistic low-contrast atmospheric landscape (foggy mountains/sky).
    Range is compressed around mid-tones (80 - 150).
    """
    w, h = 320, 240
    # Gradient background (foggy dawn)
    y_coords = np.linspace(0, 1, h)[:, None]
    x_coords = np.linspace(0, 1, w)[None, :]
    
    # Base sky gradient compressed between 100 and 150
    sky = 105 + 35 * y_coords + 10 * np.sin(x_coords * np.pi * 2)
    
    # Mountain layer 1
    mount_1 = 90 + 20 * np.sin(x_coords * 3) + 15 * np.cos(x_coords * 7)
    mount_1_2d = np.broadcast_to(mount_1, (h, w))
    mask_1 = (y_coords * h) > (h * 0.45 + mount_1)
    
    # Mountain layer 2
    mount_2 = 75 + 25 * np.sin(x_coords * 5 + 1) + 10 * np.sin(x_coords * 11)
    mount_2_2d = np.broadcast_to(mount_2, (h, w))
    mask_2 = (y_coords * h) > (h * 0.65 + mount_2)
    
    img_gray = sky.copy()
    img_gray[mask_1] = mount_1_2d[mask_1]
    img_gray[mask_2] = mount_2_2d[mask_2]

    # Add gentle color tint (misty blue-gray)
    r = np.clip(img_gray * 0.95, 0, 255).astype(np.uint8)
    g = np.clip(img_gray * 1.0, 0, 255).astype(np.uint8)
    b = np.clip(img_gray * 1.1, 0, 255).astype(np.uint8)
    
    return np.stack([r, g, b], axis=-1)


def generate_sample_high_contrast() -> np.ndarray:
    """
    Generates a vibrant, high-contrast scene with deep shadows and bright highlights
    (e.g., sunset horizon over ocean). Ideal as a target image for histogram matching.
    """
    w, h = 320, 240
    y_coords = np.linspace(0, 1, h)[:, None]
    x_coords = np.linspace(0, 1, w)[None, :]

    # Bright setting sun in the upper center
    dist_to_sun = np.sqrt((x_coords - 0.5)**2 + (y_coords - 0.35)**2 * 2.0)
    sun_glow = np.clip(1.0 - dist_to_sun * 2.5, 0, 1)

    # Ocean water in bottom half
    is_water = np.broadcast_to(y_coords > 0.6, (h, w))
    
    # Sky colors
    r = 255.0 * (1.0 - y_coords * 0.5) + sun_glow * 200.0
    g = 140.0 * (1.0 - y_coords * 0.7) + sun_glow * 180.0
    b = 40.0 * (1.0 - y_coords * 0.9) + 50.0 * (1.0 - sun_glow)

    r = np.broadcast_to(r, (h, w)).copy()
    g = np.broadcast_to(g, (h, w)).copy()
    b = np.broadcast_to(b, (h, w)).copy()

    # Water reflection
    r[is_water] = 40.0 + 120.0 * sun_glow[is_water]
    g[is_water] = 30.0 + 80.0 * sun_glow[is_water]
    b[is_water] = 60.0 + 40.0 * sun_glow[is_water]

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def generate_sample_portrait() -> np.ndarray:
    """
    Generates a smooth geometric portrait-style scene with varied tones and textures.
    """
    img = Image.new("RGB", (320, 240), color=(50, 60, 80))
    draw = ImageDraw.Draw(img)
    
    # Soft background gradient circles
    for r in range(160, 20, -20):
        draw.ellipse([160 - r, 120 - r, 160 + r, 120 + r], fill=(50 + r//2, 70 + r//3, 90 + r//4))
        
    # Central subject silhouette / face oval
    draw.ellipse([110, 50, 210, 190], fill=(210, 160, 130))
    # Eyes
    draw.ellipse([130, 100, 145, 115], fill=(40, 30, 30))
    draw.ellipse([175, 100, 190, 115], fill=(40, 30, 30))
    # Highlight
    draw.ellipse([135, 103, 140, 108], fill=(255, 255, 255))
    draw.ellipse([180, 103, 185, 108], fill=(255, 255, 255))
    # Smile
    draw.arc([140, 130, 180, 160], start=10, end=170, fill=(150, 50, 50), width=4)
    
    # Blur slightly for smooth gradients
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(img)
