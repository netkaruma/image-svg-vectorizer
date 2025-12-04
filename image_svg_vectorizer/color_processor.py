"""
Module for processing colors and contours.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from .utils import ColorInfo, bgr_to_rgb, bgr_to_hex


class ColorProcessor:
    """Class for processing image colors and contours."""
    
    def __init__(self, 
                 simplify_contours: bool = True,
                 contour_tolerance: float = 1.5,
                 min_contour_area: int = 10,
                 preserve_transparency: bool = True):
        """
        Initialize color processor.
        
        Args:
            simplify_contours: Whether to simplify contours
            contour_tolerance: Tolerance for contour simplification
            min_contour_area: Minimum contour area for processing
            preserve_transparency: Whether to preserve transparency
        """
        self.simplify_contours = simplify_contours
        self.contour_tolerance = contour_tolerance
        self.min_contour_area = min_contour_area
        self.preserve_transparency = preserve_transparency
    
    def simplify_with_kmeans(self, 
                           image: np.ndarray, 
                           num_colors: int) -> np.ndarray:
        """
        Simplify image using K-means.
        
        Args:
            image: Input BGR image
            num_colors: Number of colors for simplification
            
        Returns:
            Simplified image
        """
        # If image has transparency, process only opaque pixels
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Separate color and alpha channel
            color_image = image[:, :, :3]
            alpha_channel = image[:, :, 3]
            
            # Mask of opaque pixels
            opaque_mask = alpha_channel > 0
            
            if np.any(opaque_mask):
                # Take only opaque pixels for K-means
                opaque_pixels = color_image[opaque_mask]
                pixel_values = opaque_pixels.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)
                
                if len(pixel_values) > num_colors:
                    # Criteria for K-means
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                    
                    # Perform K-means
                    _, labels, centers = cv2.kmeans(
                        pixel_values,
                        min(num_colors, len(pixel_values)),
                        None,
                        criteria,
                        10,
                        cv2.KMEANS_RANDOM_CENTERS
                    )
                    
                    # Convert centers to 8-bit integers
                    centers = np.uint8(centers)
                    
                    # Create simplified image
                    simplified_color = color_image.copy()
                    simplified_color[opaque_mask] = centers[labels.flatten()]
                    
                    # Combine with alpha channel
                    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                    result[:, :, :3] = simplified_color
                    result[:, :, 3] = alpha_channel
                    return result[:, :, :3]  # Return only color for further processing
            else:
                # All pixels are transparent
                return np.zeros_like(color_image)
        else:
            # Standard processing without transparency
            pixel_values = image.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # Criteria for K-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Perform K-means
            _, labels, centers = cv2.kmeans(
                pixel_values,
                num_colors,
                None,
                criteria,
                10,
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Convert centers to 8-bit integers
            centers = np.uint8(centers)
            
            # Create simplified image
            simplified_image = centers[labels.flatten()]
            simplified_image = simplified_image.reshape(image.shape)
            
            return simplified_image
    
    def simplify_with_median_cut(self, 
                                image: np.ndarray, 
                                num_colors: int) -> np.ndarray:
        """
        Alternative method: median cut (faster).
        
        Args:
            image: Input image
            num_colors: Number of colors
            
        Returns:
            Simplified image
        """
        # If image has transparency, process only opaque pixels
        if len(image.shape) > 2 and image.shape[2] == 4:
            color_image = image[:, :, :3]
            alpha_channel = image[:, :, 3]
            opaque_mask = alpha_channel > 0
            
            if np.any(opaque_mask):
                # Take only opaque pixels
                opaque_pixels = color_image[opaque_mask]
            else:
                # All pixels are transparent
                return np.zeros_like(color_image)
        else:
            color_image = image
            opaque_pixels = color_image.reshape(-1, 3)
            opaque_mask = np.ones(color_image.shape[:2], dtype=bool)
        
        # Convert to list of pixels
        pixels = opaque_pixels.tolist()
        
        # Median cut implementation
        def median_cut(pixel_array, num_colors):
            if len(pixel_array) == 0 or num_colors <= 1:
                return [np.mean(pixel_array, axis=0) if len(pixel_array) > 0 else [0, 0, 0]]
            
            # Find channel with greatest spread
            r_range = np.ptp(pixel_array[:, 0])
            g_range = np.ptp(pixel_array[:, 1])
            b_range = np.ptp(pixel_array[:, 2])
            
            channel = np.argmax([r_range, g_range, b_range])
            
            # Sort by selected channel
            sorted_pixels = pixel_array[pixel_array[:, channel].argsort()]
            
            # Split by median
            median_index = len(sorted_pixels) // 2
            
            # Apply recursively to both halves
            left = median_cut(sorted_pixels[:median_index], num_colors // 2)
            right = median_cut(sorted_pixels[median_index:], num_colors // 2)
            
            return left + right
        
        pixel_array = np.array(pixels)
        color_palette = median_cut(pixel_array, num_colors)
        
        # Quantize image
        quantized = np.zeros_like(color_image)
        for i in range(color_image.shape[0]):
            for j in range(color_image.shape[1]):
                if opaque_mask[i, j]:
                    pixel = color_image[i, j]
                    # Find closest color in palette
                    distances = [np.linalg.norm(pixel - color) for color in color_palette]
                    closest_idx = np.argmin(distances)
                    quantized[i, j] = color_palette[closest_idx]
        
        # Restore alpha channel if needed
        if len(image.shape) > 2 and image.shape[2] == 4:
            result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            result[:, :, :3] = quantized
            result[:, :, 3] = image[:, :, 3]
            return result[:, :, :3]  # Return only color
        
        return quantized
    
    def find_color_contours(self, 
                          image: np.ndarray, 
                          color_bgr: Tuple[int, int, int],
                          alpha_channel: Optional[np.ndarray] = None,
                          min_alpha: int = 1) -> Optional[ColorInfo]:
        """
        Find contours for specific color considering transparency.
        
        Args:
            image: Simplified image
            color_bgr: Color in BGR format
            alpha_channel: Image alpha channel
            min_alpha: Minimum alpha value to consider
            
        Returns:
            ColorInfo object or None
        """
        # Create mask for color
        mask = cv2.inRange(image, color_bgr, color_bgr)
        
        # Consider transparency if available
        if alpha_channel is not None and self.preserve_transparency:
            # Mask for sufficient transparency
            alpha_mask = alpha_channel >= min_alpha
            # Combine color and transparency masks
            mask = cv2.bitwise_and(mask, mask, mask=alpha_mask.astype(np.uint8))
        
        # Skip too small areas
        if np.sum(mask > 0) < self.min_contour_area:
            return None
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # Separate into external and internal contours
        external_contours = []
        internal_contours = []
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, contour in enumerate(contours):
                # Skip too small contours
                if cv2.contourArea(contour) < 5:
                    continue
                    
                if hierarchy[i][3] == -1:  # External contour
                    external_contours.append(contour)
                else:  # Internal contour
                    internal_contours.append(contour)
        else:
            external_contours = contours
        
        # Calculate perimeters
        external_perimeters = [cv2.arcLength(c, True) for c in external_contours]
        internal_perimeters = [cv2.arcLength(c, True) for c in internal_contours]
        
        # Determine transparent areas inside the color
        transparent_areas = None
        if alpha_channel is not None and self.preserve_transparency:
            # Find areas with different transparency inside this color
            color_alpha = alpha_channel.copy()
            color_alpha[mask == 0] = 0  # Zero alpha outside this color
            transparent_areas = color_alpha < 255
        
        # Create ColorInfo object
        color_info = ColorInfo(
            bgr=color_bgr,
            rgb=bgr_to_rgb(color_bgr),
            hex=bgr_to_hex(color_bgr),
            area=np.sum(mask > 0),
            contours=contours,
            external_contours=external_contours,
            internal_contours=internal_contours,
            external_perimeters=external_perimeters,
            internal_perimeters=internal_perimeters,
            mask=mask,
            transparent_areas=transparent_areas
        )
        
        return color_info
    
    def contours_to_svg_path(self, 
                            contours: List[np.ndarray],
                            simplify: bool = None,
                            tolerance: float = None) -> List[str]:
        """
        Convert contours to SVG path strings.
        
        Args:
            contours: List of OpenCV contours
            simplify: Whether to simplify contours (if None, uses self.simplify_contours)
            tolerance: Simplification tolerance (if None, uses self.contour_tolerance)
            
        Returns:
            List of SVG path strings
        """
        if simplify is None:
            simplify = self.simplify_contours
        if tolerance is None:
            tolerance = self.contour_tolerance
        
        paths = []
        
        for contour in contours:
            if len(contour) < 2:
                continue
            
            # Simplify contour if needed
            if simplify:
                epsilon = tolerance
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Create SVG path
            path = "M "
            for i, point in enumerate(contour):
                x, y = point[0]
                path += f"{x} {y}"
                if i < len(contour) - 1:
                    path += " L "
            
            # Close contour
            if len(contour) > 2:
                path += " Z"
            paths.append(path)
        
        return paths