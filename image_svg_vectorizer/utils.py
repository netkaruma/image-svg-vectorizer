"""
Utility classes and functions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore')

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
    simplified_image_with_alpha: Optional[np.ndarray] = None  # Moved to the end


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


def calculate_edge_density(image: np.ndarray) -> float:
    """
    Calculates edge density to determine image complexity.
    
    Args:
        image: BGR image
        
    Returns:
        Edge density (0-1)
    """
    if len(image.shape) == 2:  # Grayscale
        gray = image
    elif image.shape[2] == 4:  # RGBA
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:  # BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    return edge_density


def auto_determine_colors(image: np.ndarray, max_colors: int = 16) -> int:
    """
    Automatically determines optimal number of colors.
    
    Args:
        image: BGR image
        max_colors: Maximum number of colors
        
    Returns:
        Optimal number of colors
    """
    edge_density = calculate_edge_density(image)
    
    # Increase color count if there is transparency
    transparency_factor = 1.0
    if len(image.shape) > 2 and image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        transparency_complexity = np.std(alpha_channel) / 255.0
        transparency_factor = 1.0 + transparency_complexity * 0.5
    
    if edge_density < 0.01:  # Very simple
        return max(4, int(4 * transparency_factor))
    elif edge_density < 0.05:  # Simple
        return max(6, int(6 * transparency_factor))
    elif edge_density < 0.1:   # Medium complexity
        return max(8, int(8 * transparency_factor))
    else:                      # Complex
        return min(int(12 * transparency_factor), max_colors)