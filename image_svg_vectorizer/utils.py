"""
Utility classes and functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
import numpy as np
import warnings
from enum import Enum

warnings.filterwarnings('ignore')

class ColorMode(Enum):
    """Mode for color count determination."""
    AUTO = "auto"
    MANUAL = "manual"


@dataclass
class ColorInfo:
    """Class for storing color information."""
    bgr: Tuple[int, int, int]
    rgb: Tuple[int, int, int]
    hex: str
    area: int
    contours: List[np.ndarray]
    external_contours: List[np.ndarray]
    internal_contours: List[np.ndarray]
    external_perimeters: List[float]
    internal_perimeters: List[float]
    mask: np.ndarray
    alpha: int = 255
    has_transparency: bool = False
    transparent_areas: Optional[np.ndarray] = None


@dataclass
class ProcessingResults:
    """Class for storing image processing results."""
    original_image: np.ndarray
    simplified_image: np.ndarray
    unique_colors: np.ndarray
    color_info: Dict[str, ColorInfo]
    image_shape: Tuple[int, int, int]
    processing_time: float = 0.0
    transparency_info: Optional[Dict] = None
    simplified_image_with_alpha: Optional[np.ndarray] = None
    color_mode: ColorMode = ColorMode.AUTO
    num_colors_used: int = 0


def bgr_to_rgb(bgr_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Converts color from BGR to RGB."""
    return tuple(reversed(bgr_color))


def bgr_to_hex(bgr_color: Tuple[int, int, int]) -> str:
    """Converts color from BGR to HEX."""
    rgb = bgr_to_rgb(bgr_color)
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def rgba_to_hex(rgba_color: Tuple[int, int, int, int]) -> str:
    """Converts color from RGBA to HEX with alpha channel."""
    if len(rgba_color) == 4:
        return '#{:02x}{:02x}{:02x}{:02x}'.format(*rgba_color)
    else:
        return bgr_to_hex(rgba_color[:3])


def hex_to_rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Converts HEX color to RGBA."""
    hex_color = hex_color.lstrip('#')
    
    if len(hex_color) == 8:  # With alpha channel
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
            int(hex_color[6:8], 16)
        )
    elif len(hex_color) == 6:  # Without alpha channel
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
            255
        )
    else:
        raise ValueError(f"Invalid HEX color format: {hex_color}")


def calculate_image_complexity(image: np.ndarray) -> Dict[str, float]:
    """
    Calculates various image complexity metrics.
    
    Args:
        image: BGR image
        
    Returns:
        Dictionary with complexity metrics
    """
    if len(image.shape) == 2:  # Grayscale
        gray = image
    elif image.shape[2] == 4:  # RGBA
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:  # BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    
    # Color variance
    if len(image.shape) > 2:
        color_channels = image[:, :, :3] if image.shape[2] == 4 else image
        color_variance = np.mean([np.std(channel) for channel in cv2.split(color_channels)]) / 255.0
    else:
        color_variance = np.std(gray) / 255.0
    
    # Texture complexity using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_complexity = np.var(laplacian) / 1000.0  # Normalized
    
    # Histogram entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10)) / 8.0  # Normalized to 0-1
    
    return {
        'edge_density': edge_density,
        'color_variance': color_variance,
        'texture_complexity': min(texture_complexity, 1.0),
        'entropy': entropy,
        'overall_complexity': (edge_density + color_variance + texture_complexity + entropy) / 4.0
    }


def auto_determine_colors(image: np.ndarray, max_colors: int = 64, min_colors: int = 4) -> int:
    """
    Automatically determines optimal number of colors based on image complexity.
    
    Args:
        image: BGR image
        max_colors: Maximum number of colors
        min_colors: Minimum number of colors
        
    Returns:
        Optimal number of colors
    """
    # Calculate complexity metrics
    complexity = calculate_image_complexity(image)
    overall = complexity['overall_complexity']
    
    # Base calculation with logarithmic scaling for better distribution
    # This creates a smoother, more gradual increase in color count
    if overall < 0.1:  # Very simple (logos, icons)
        base_colors = 4
    elif overall < 0.3:  # Simple (diagrams, cartoons)
        base_colors = 6
    elif overall < 0.5:  # Medium (simple illustrations)
        base_colors = 8
    elif overall < 0.7:  # Complex (detailed illustrations)
        base_colors = 12
    else:  # Very complex (photos, detailed artwork)
        base_colors = 16
    
    # Adjust for image size
    h, w = image.shape[:2]
    total_pixels = h * w
    size_factor = np.log10(total_pixels / 10000) / 3  # Normalized size factor
    size_factor = np.clip(size_factor, -0.5, 1.0)  # Limit influence
    
    # Adjust for transparency complexity if present
    transparency_factor = 1.0
    if len(image.shape) > 2 and image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        if np.any(alpha_channel < 255):
            # Measure transparency complexity
            alpha_non_binary = alpha_channel[(alpha_channel > 0) & (alpha_channel < 255)]
            if len(alpha_non_binary) > 0:
                transparency_complexity = np.std(alpha_non_binary) / 128.0
                transparency_factor = 1.0 + transparency_complexity * 0.3
    
    # Calculate final color count with smooth scaling
    color_count = int(base_colors * (1 + size_factor) * transparency_factor)
    
    # Apply logarithmic scaling for high complexity images
    if overall > 0.7:
        # Add extra colors for high complexity, but with diminishing returns
        extra_colors = int((overall - 0.7) * 20)
        color_count += extra_colors
    
    # Ensure within bounds
    color_count = max(min_colors, min(color_count, max_colors))
    
    # Round to even number for better k-means performance
    color_count = (color_count + 1) // 2 * 2
    
    # Debug output
    print(f"Auto colors: complexity={overall:.3f}, size={w}x{h}, "
          f"base={base_colors}, size_factor={size_factor:.2f}, "
          f"transparency_factor={transparency_factor:.2f}, result={color_count}")
    
    return color_count
