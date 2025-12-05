"""
Main image vectorizer module with acceleration support.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from .utils import ProcessingResults, ColorInfo, ColorMode, auto_determine_colors, bgr_to_rgb, bgr_to_hex, calculate_image_complexity
from .color_processor import ColorProcessor
from .svg_exporter import SVGExporter


class ColorImageVectorizer:
    """
    Main class for vectorizing color images with acceleration support.
    
    Usage example:
    >>> vectorizer = ColorImageVectorizer()
    >>> results = vectorizer.process_image("input.png", num_colors=8)
    >>> vectorizer.create_svg(results, "output.svg")
    """
    
    def __init__(self, 
                 simplify_contours: bool = True,
                 contour_tolerance: float = 1.5,
                 min_contour_area: int = 10,
                 preserve_transparency: bool = True,
                 use_acceleration: bool = True,
                 expand_contours: bool = True,
                 expansion_pixels: int = 1,
                 color_mode: ColorMode = ColorMode.AUTO,
                 num_colors: Optional[int] = None):
        """
        Initialize the vectorizer with optional acceleration.
        
        Args:
            simplify_contours: Whether to simplify contours
            contour_tolerance: Tolerance for contour simplification
            min_contour_area: Minimum contour area for processing
            preserve_transparency: Whether to preserve transparency (alpha channel)
            use_acceleration: Use CPU/GPU acceleration (Numba)
            expand_contours: Whether to expand contours for better edge handling
            expansion_pixels: Number of pixels to expand contours
            color_mode: Color determination mode (AUTO or MANUAL)
            num_colors: Manual number of colors (overrides auto mode)
        """
        self.preserve_transparency = preserve_transparency
        self.use_acceleration = use_acceleration
        self.color_mode = color_mode
        
        # If num_colors is explicitly provided, switch to MANUAL mode
        if num_colors is not None:
            self.color_mode = ColorMode.MANUAL
            self.num_colors = num_colors
            print(f"✓ Manual color mode with {num_colors} colors")
        else:
            self.num_colors = None
            print(f"✓ Auto color mode enabled")
        
        # Check if Numba is available for acceleration
        try:
            import numba
            self.numba_available = True
            if use_acceleration:
                print("✓ Numba acceleration enabled")
        except ImportError:
            self.numba_available = False
            if use_acceleration:
                print("⚠ Numba not installed. Install: pip install numba")
                print("  Using standard NumPy processing")
        
        self.color_processor = ColorProcessor(
            simplify_contours=simplify_contours,
            contour_tolerance=contour_tolerance,
            min_contour_area=min_contour_area,
            preserve_transparency=preserve_transparency,
            expand_contours=expand_contours,
            expansion_pixels=expansion_pixels,
        )
        self.svg_exporter = SVGExporter(self.color_processor)
    
    def set_color_mode(self, mode: ColorMode, num_colors: Optional[int] = None):
        """
        Set the color mode and optionally the number of colors.
        
        Args:
            mode: ColorMode.AUTO or ColorMode.MANUAL
            num_colors: Required if mode is MANUAL
        """
        self.color_mode = mode
        if mode == ColorMode.MANUAL:
            if num_colors is None:
                raise ValueError("num_colors must be provided for MANUAL mode")
            self.num_colors = num_colors
            print(f"Switched to MANUAL mode with {num_colors} colors")
        else:
            self.num_colors = None
            print("Switched to AUTO mode")
    
    def _kmeans_accelerated(self, 
                           pixels: np.ndarray, 
                           num_colors: int,
                           max_iterations: int = 20,
                           epsilon: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Accelerated K-means using Numba or vectorized NumPy.
        
        Args:
            pixels: Pixel array (N, 3)
            num_colors: Number of colors
            max_iterations: Maximum iterations
            epsilon: Convergence threshold
            
        Returns:
            Centroids and labels
        """
        n_pixels = pixels.shape[0]
        pixels_float = pixels.astype(np.float32)
        
        # Initialize centroids with random pixels
        np.random.seed(42)
        idx = np.random.choice(n_pixels, num_colors, replace=False)
        centroids = pixels_float[idx].copy()
        
        if self.use_acceleration and self.numba_available:
            # Use Numba-accelerated version
            centroids, labels = self._kmeans_numba(pixels_float, centroids, num_colors, max_iterations)
        else:
            # Use vectorized NumPy version
            centroids, labels = self._kmeans_vectorized(pixels_float, centroids, num_colors, max_iterations)
        
        return centroids.astype(np.uint8), labels
    
    def _kmeans_numba(self, 
                     pixels: np.ndarray,
                     centroids: np.ndarray,
                     num_colors: int,
                     max_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated K-means."""
        import numba
        
        @numba.jit(nopython=True, parallel=True)
        def kmeans_core(pixels, centroids, num_colors, max_iterations):
            n_pixels = pixels.shape[0]
            labels = np.zeros(n_pixels, dtype=np.int32)
            
            for iteration in range(max_iterations):
                # Assignment step (parallel)
                for i in numba.prange(n_pixels):
                    min_dist = 1e9
                    best_k = 0
                    
                    for k in range(num_colors):
                        dist = 0.0
                        # Calculate Euclidean distance
                        for c in range(3):
                            diff = pixels[i, c] - centroids[k, c]
                            dist += diff * diff
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_k = k
                    
                    labels[i] = best_k
                
                # Update centroids
                new_centroids = np.zeros_like(centroids)
                counts = np.zeros(num_colors, dtype=np.float32)
                
                for i in range(n_pixels):
                    k = labels[i]
                    for c in range(3):
                        new_centroids[k, c] += pixels[i, c]
                    counts[k] += 1
                
                # Normalize and handle empty clusters
                changed = False
                for k in range(num_colors):
                    if counts[k] > 0:
                        old_centroid = centroids[k].copy()
                        new_centroids[k] /= counts[k]
                        
                        # Check if centroid changed significantly
                        for c in range(3):
                            if abs(new_centroids[k, c] - old_centroid[c]) > 0.1:
                                changed = True
                    else:
                        # Reinitialize empty cluster
                        idx = np.random.randint(0, n_pixels)
                        new_centroids[k] = pixels[idx]
                        changed = True
                
                centroids = new_centroids.copy()
                
                if not changed:
                    break
            
            return centroids, labels
        
        return kmeans_core(pixels, centroids, num_colors, max_iterations)
    
    def _kmeans_vectorized(self,
                          pixels: np.ndarray,
                          centroids: np.ndarray,
                          num_colors: int,
                          max_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized K-means using NumPy."""
        n_pixels = pixels.shape[0]
        
        for iteration in range(max_iterations):
            # Vectorized distance calculation
            # Reshape for broadcasting: (n_pixels, 1, 3) - (1, num_colors, 3)
            diff = pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            distances = np.sum(diff * diff, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(num_colors):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = np.mean(pixels[mask], axis=0)
                else:
                    # Reinitialize empty cluster
                    new_centroids[k] = pixels[np.random.randint(0, n_pixels)]
            
            # Check convergence
            if np.allclose(centroids, new_centroids, rtol=0.01):
                centroids = new_centroids
                break
            
            centroids = new_centroids
        
        return centroids, labels
    
    def _simplify_with_kmeans_accelerated(self, 
                                         image: np.ndarray, 
                                         num_colors: int) -> np.ndarray:
        """
        Accelerated K-means color simplification.
        
        Args:
            image: Input BGR image
            num_colors: Number of colors
            
        Returns:
            Simplified image
        """
        start_time = time.time()
        
        # Handle transparency
        if len(image.shape) > 2 and image.shape[2] == 4:
            color_image = image[:, :, :3]
            alpha_channel = image[:, :, 3]
            opaque_mask = alpha_channel > 0
            
            if np.any(opaque_mask):
                opaque_pixels = color_image[opaque_mask]
                num_colors = min(num_colors, len(opaque_pixels))
                
                if num_colors <= 1:
                    # Special case: single color
                    simplified_color = color_image.copy()
                    if len(opaque_pixels) > 0:
                        avg_color = np.mean(opaque_pixels, axis=0)
                        simplified_color[opaque_mask] = avg_color
                else:
                    centroids, labels = self._kmeans_accelerated(opaque_pixels, num_colors)
                    simplified_color = color_image.copy()
                    simplified_color[opaque_mask] = centroids[labels]
                
                result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                result[:, :, :3] = simplified_color
                result[:, :, 3] = alpha_channel
                print(f"K-means with transparency: {time.time() - start_time:.2f}s")
                return result[:, :, :3]
            else:
                # All pixels are transparent
                return np.zeros_like(color_image)
        else:
            # Standard image without alpha
            h, w, c = image.shape
            pixels = image.reshape(-1, 3)
            
            if num_colors <= 1:
                # Single color case
                avg_color = np.mean(pixels, axis=0)
                simplified = np.full((h, w, 3), avg_color, dtype=np.uint8)
            else:
                centroids, labels = self._kmeans_accelerated(pixels, num_colors)
                simplified = centroids[labels].reshape(h, w, c)
            
            print(f"K-means: {time.time() - start_time:.2f}s")
            return simplified
    
    def _median_cut_accelerated(self, 
                               image: np.ndarray, 
                               num_colors: int) -> np.ndarray:
        """
        Accelerated median cut color quantization.
        
        Args:
            image: Input image
            num_colors: Number of colors
            
        Returns:
            Simplified image
        """
        start_time = time.time()
        
        if len(image.shape) > 2 and image.shape[2] == 4:
            color_image = image[:, :, :3]
            alpha_channel = image[:, :, 3]
            opaque_mask = alpha_channel > 0
            
            if np.any(opaque_mask):
                opaque_pixels = color_image[opaque_mask]
            else:
                return np.zeros_like(color_image)
        else:
            color_image = image
            opaque_pixels = color_image.reshape(-1, 3)
            opaque_mask = np.ones(color_image.shape[:2], dtype=bool)
        
        # Vectorized median cut implementation
        def median_cut_vectorized(pixel_array, num_colors):
            if len(pixel_array) == 0 or num_colors <= 1:
                if len(pixel_array) == 0:
                    return [np.zeros(3, dtype=np.uint8)]
                else:
                    return [np.mean(pixel_array, axis=0).astype(np.uint8)]
            
            # Find channel with greatest range
            ranges = np.ptp(pixel_array, axis=0)
            channel = np.argmax(ranges)
            
            # Sort by selected channel
            sorted_idx = np.argsort(pixel_array[:, channel])
            sorted_pixels = pixel_array[sorted_idx]
            
            # Split by median
            median_idx = len(sorted_pixels) // 2
            
            # Recursive processing
            left = median_cut_vectorized(sorted_pixels[:median_idx], num_colors // 2)
            right = median_cut_vectorized(sorted_pixels[median_idx:], num_colors // 2)
            
            return left + right
        
        # Generate palette
        palette = median_cut_vectorized(opaque_pixels, num_colors)
        palette_array = np.array(palette, dtype=np.uint8)
        
        # Vectorized quantization
        # Reshape for broadcasting
        pixels_reshaped = opaque_pixels.reshape(-1, 1, 3)
        palette_reshaped = palette_array.reshape(1, -1, 3)
        
        # Calculate distances
        diff = pixels_reshaped - palette_reshaped
        distances = np.sum(diff * diff, axis=2)
        closest_idx = np.argmin(distances, axis=1)
        
        # Create result
        if len(image.shape) > 2 and image.shape[2] == 4:
            quantized = np.zeros_like(color_image)
            quantized[opaque_mask] = palette_array[closest_idx]
            
            result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            result[:, :, :3] = quantized
            result[:, :, 3] = alpha_channel
            result_image = result[:, :, :3]
        else:
            quantized = palette_array[closest_idx].reshape(color_image.shape)
            result_image = quantized
        
        print(f"Median cut: {time.time() - start_time:.2f}s")
        return result_image
    
    def process_image(self, 
                    image_path: Union[str, Path],
                    num_colors: Optional[int] = None,
                    method: str = 'kmeans',
                    max_colors: int = 64,
                    min_colors: int = 4) -> ProcessingResults:
        """
        Main method for processing an image with acceleration support.
        
        Args:
            image_path: Path to the image
            num_colors: Number of colors for simplification (overrides auto mode)
            method: Simplification method ('kmeans' or 'median')
            max_colors: Maximum number of colors for auto mode
            min_colors: Minimum number of colors for auto mode
            
        Returns:
            ProcessingResults object
        """
        total_start_time = time.time()
        
        # Load image with alpha channel preservation
        if isinstance(image_path, (str, Path)):
            # Load as-is (with alpha channel if present)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
        else:
            # Assume it's already a numpy array
            image = image_path
        
        # Determine color mode and count
        if num_colors is not None:
            # Override with provided num_colors
            color_mode = ColorMode.MANUAL
            colors_to_use = num_colors
            print(f"Using manual color count: {colors_to_use}")
        elif self.color_mode == ColorMode.MANUAL and self.num_colors is not None:
            # Use preset manual value
            color_mode = ColorMode.MANUAL
            colors_to_use = self.num_colors
            print(f"Using preset manual color count: {colors_to_use}")
        else:
            # Auto mode
            color_mode = ColorMode.AUTO
            colors_to_use = auto_determine_colors(image, max_colors, min_colors)
            print(f"Auto-determined color count: {colors_to_use}")
        
        # Check for alpha channel presence
        has_alpha = image.shape[2] == 4 if len(image.shape) > 2 else False
        
        if has_alpha and self.preserve_transparency:
            print(f"Alpha channel detected. Preserving transparency...")
            # Separate color and alpha channel
            alpha_channel = image[:, :, 3]
            color_image = image[:, :, :3]
            
            # Create transparency mask
            transparency_mask = (alpha_channel > 0).astype(np.uint8) * 255
            
            # Find fully transparent areas
            fully_transparent = alpha_channel == 0
            partially_transparent = (alpha_channel > 0) & (alpha_channel < 255)
            
            # Store transparency information
            transparency_info = {
                'has_alpha': True,
                'alpha_channel': alpha_channel,
                'transparency_mask': transparency_mask,
                'fully_transparent': fully_transparent,
                'partially_transparent': partially_transparent,
                'max_alpha': np.max(alpha_channel) if np.any(alpha_channel > 0) else 255
            }
        else:
            if len(image.shape) == 2:  # Grayscale
                color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4 and not self.preserve_transparency:
                color_image = image[:, :, :3]
            else:
                color_image = image.copy()
            transparency_info = {'has_alpha': False}
        
        # Simplify colors with acceleration
        if method == 'median':
            simplified_image = self._median_cut_accelerated(color_image, colors_to_use)
        else:  # kmeans by default
            simplified_image = self._simplify_with_kmeans_accelerated(color_image, colors_to_use)
        
        # Restore alpha channel if needed
        if has_alpha and self.preserve_transparency:
            # Create image with alpha channel for preview
            result_image = np.zeros((simplified_image.shape[0], simplified_image.shape[1], 4), dtype=np.uint8)
            result_image[:, :, :3] = simplified_image
            result_image[:, :, 3] = transparency_mask
            simplified_image_with_alpha = result_image
            
            # Find unique colors considering transparency
            # Create array of all pixels with alpha channel
            h, w = simplified_image.shape[:2]
            pixels_with_alpha = np.zeros((h * w, 4), dtype=np.uint8)
            
            # Optimized way to get unique colors with transparency
            # Process in chunks to avoid memory issues with large images
            chunk_size = 1000
            unique_colors_set = set()
            
            for i in range(0, h, chunk_size):
                i_end = min(i + chunk_size, h)
                for j in range(0, w, chunk_size):
                    j_end = min(j + chunk_size, w)
                    
                    chunk = simplified_image[i:i_end, j:j_end]
                    alpha_chunk = transparency_mask[i:i_end, j:j_end]
                    
                    # Combine color and alpha
                    chunk_h, chunk_w = chunk.shape[:2]
                    for ii in range(chunk_h):
                        for jj in range(chunk_w):
                            color_tuple = tuple(chunk[ii, jj])
                            alpha_val = alpha_chunk[ii, jj]
                            unique_colors_set.add((*color_tuple, alpha_val))
            
            unique_colors = np.array(list(unique_colors_set), dtype=np.uint8)
            print(f"Found {len(unique_colors)} unique color+transparency combinations")
        else:
            simplified_image_with_alpha = None
            # Optimized unique colors without transparency
            unique_colors = np.unique(simplified_image.reshape(-1, 3), axis=0)
        
        # Process each color
        color_info_dict = {}
        
        if has_alpha and self.preserve_transparency:
            # Process colors considering transparency
            for color_data in unique_colors:
                color_bgr = tuple(map(int, color_data[:3]))
                alpha_value = int(color_data[3])
                
                # Skip fully transparent "colors" (alpha=0)
                # They will be represented as absence of shapes
                if alpha_value == 0:
                    continue
                    
                # Find contours only for pixels with sufficient transparency
                color_info = self.color_processor.find_color_contours(
                    simplified_image, 
                    color_bgr,
                    alpha_channel=transparency_info['alpha_channel'],
                    min_alpha=1  # Minimum alpha value to consider
                )
                
                if color_info is not None:
                    # Add transparency information
                    color_info.alpha = alpha_value
                    color_info.has_transparency = alpha_value < 255
                    
                    # Create HEX with alpha channel
                    hex_with_alpha = bgr_to_hex(color_bgr) + f"{alpha_value:02x}"
                    color_info_dict[hex_with_alpha] = color_info
        else:
            # Process without considering transparency
            for color in unique_colors:
                color_bgr = tuple(map(int, color))
                color_info = self.color_processor.find_color_contours(simplified_image, color_bgr)
                
                if color_info is not None:
                    color_info.alpha = 255
                    color_info.has_transparency = False
                    color_info_dict[color_info.hex] = color_info
        
        # Sort colors by area (largest to smallest)
        sorted_colors = sorted(
            color_info_dict.items(),
            key=lambda x: x[1].area,
            reverse=True
        )
        color_info_dict = {k: v for k, v in sorted_colors}
        
        # Add information about fully transparent areas
        if has_alpha and self.preserve_transparency and transparency_info['fully_transparent'].any():
            # Create special object for transparent areas
            transparent_area = int(np.sum(transparency_info['fully_transparent']))
            if transparent_area > 0:
                print(f"Detected {transparent_area} pixels with full transparency")
        
        processing_time = time.time() - total_start_time
        
        # Create results object
        results = ProcessingResults(
            original_image=image,
            simplified_image=simplified_image,
            unique_colors=unique_colors,
            color_info=color_info_dict,
            image_shape=image.shape,
            processing_time=processing_time,
            transparency_info=transparency_info if has_alpha else None,
            simplified_image_with_alpha=simplified_image_with_alpha,
            color_mode=color_mode,
            num_colors_used=colors_to_use
        )
        
        # Print statistics
        print(f"Processing completed in {processing_time:.2f} sec")
        print(f"Color mode: {color_mode.value}")
        print(f"Colors used: {colors_to_use}")
        print(f"Unique colors: {len(unique_colors)}")
        print(f"Colors with contours: {len(color_info_dict)}")
        if has_alpha and self.preserve_transparency:
            h, w = simplified_image.shape[:2]
            transparent_pixels = np.sum(transparency_info['fully_transparent'])
            print(f"Transparent pixels: {transparent_pixels} ({transparent_pixels/(h*w)*100:.1f}%)")
        
        return results
    
    def create_svg(self, 
                  results: ProcessingResults,
                  output_path: Union[str, Path],
                  style: str = 'colored',
                  stroke_width: float = 0.5,
                  stroke_color: str = '#000000',
                  add_stroke: bool = False,
                  flatten_layers: bool = False,
                  preserve_transparency: bool = None) -> str:
        """
        Create SVG file from processing results.
        
        Args:
            results: Image processing results
            output_path: Path to save SVG
            style: SVG style ('colored', 'outline', 'minimal')
            stroke_width: Stroke width
            stroke_color: Stroke color
            add_stroke: Whether to add stroke
            flatten_layers: Whether to merge all layers into one
            preserve_transparency: Whether to preserve transparency
            
        Returns:
            Path to saved file
        """
        if preserve_transparency is None:
            preserve_transparency = self.preserve_transparency
            
        return self.svg_exporter.create_svg(
            results, output_path, style, stroke_width, 
            stroke_color, add_stroke, flatten_layers,
            preserve_transparency
        )
    
    def export_stats(self, 
                    results: ProcessingResults,
                    output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Export processing statistics to JSON.
        
        Args:
            results: Processing results
            output_path: Path to save JSON
            
        Returns:
            Dictionary with statistics
        """
        return self.svg_exporter.export_stats(results, output_path)
    
    def create_preview(self, 
                      results: ProcessingResults,
                      output_path: Union[str, Path] = None,
                      show: bool = True,
                      show_alpha: bool = True) -> Optional[np.ndarray]:
        """
        Create preview of processing results.
        
        Args:
            results: Processing results
            output_path: Path to save preview
            show: Whether to show preview
            show_alpha: Whether to show alpha channel
            
        Returns:
            Preview image
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        if len(results.original_image.shape) == 2:  # Grayscale
            axes[0, 0].imshow(results.original_image, cmap='gray')
        elif results.original_image.shape[2] == 4:  # RGBA
            axes[0, 0].imshow(cv2.cvtColor(results.original_image, cv2.COLOR_BGRA2RGBA))
        else:  # BGR
            axes[0, 0].imshow(cv2.cvtColor(results.original_image, cv2.COLOR_BGR2RGB))
        
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Simplified image
        if results.simplified_image_with_alpha is not None and show_alpha:
            axes[0, 1].imshow(cv2.cvtColor(results.simplified_image_with_alpha, cv2.COLOR_BGRA2RGBA))
            title = f'Simplified ({len(results.unique_colors)} colors with transparency)'
        else:
            axes[0, 1].imshow(cv2.cvtColor(results.simplified_image, cv2.COLOR_BGR2RGB))
            title = f'Simplified ({len(results.unique_colors)} colors)'
        
        # Add color mode info to title
        if results.color_mode == ColorMode.AUTO:
            title += f'\nAuto mode, {results.num_colors_used} colors used'
        else:
            title += f'\nManual mode, {results.num_colors_used} colors used'
        
        axes[0, 1].set_title(title)
        axes[0, 1].axis('off')
        
        # Color palette with transparency
        palette_height = 50
        palette_width = 300
        palette = np.zeros((palette_height, palette_width, 4), dtype=np.uint8)
        
        colors = list(results.color_info.values())
        num_colors = len(colors)
        if num_colors > 0:
            segment_width = palette_width // num_colors
            for i, color_data in enumerate(colors):
                start_x = i * segment_width
                end_x = min((i + 1) * segment_width, palette_width)
                palette[:, start_x:end_x, :3] = color_data.rgb
                palette[:, start_x:end_x, 3] = getattr(color_data, 'alpha', 255)
        
        axes[1, 0].imshow(palette)
        axes[1, 0].set_title(f'Palette ({num_colors} colors, white=transparency)')
        axes[1, 0].axis('off')
        
        # Contours (first 3 colors)
        contour_img = np.zeros((*results.original_image.shape[:2], 4), dtype=np.uint8)
        colors_to_show = min(3, len(colors))
        
        for i in range(colors_to_show):
            color_data = colors[i]
            color_bgr = color_data.bgr
            
            # Create temporary image with transparency
            temp_img = np.zeros((*results.original_image.shape[:2], 4), dtype=np.uint8)
            
            # Draw external contours
            for contour in color_data.external_contours:
                cv2.drawContours(temp_img[:, :, :3], [contour], -1, color_bgr, 2)
                cv2.drawContours(temp_img[:, :, 3], [contour], -1, 255, 2)
            
            # Draw internal contours
            for contour in color_data.internal_contours:
                cv2.drawContours(temp_img[:, :, :3], [contour], -1, 
                                tuple(int(c * 0.7) for c in color_bgr), 1)
                cv2.drawContours(temp_img[:, :, 3], [contour], -1, 255, 1)
            
            contour_img = cv2.addWeighted(contour_img, 1.0, temp_img, 1.0, 0)
        
        axes[1, 1].imshow(np.clip(cv2.cvtColor(contour_img, cv2.COLOR_BGRA2RGBA), 0, 255))
        axes[1, 1].set_title(f'Contours (first {colors_to_show} colors)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save if path is provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
            print(f"Preview saved: {output_path}")
        
        # Show if needed
        if show:
            plt.show()
        else:
            plt.close()
        
        # Convert figure to numpy image
        fig.canvas.draw()
        preview_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        preview_img = preview_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return preview_img


# Convenience functions
def vectorize_image(input_path: Union[str, Path],
                   output_path: Union[str, Path],
                   num_colors: Optional[int] = None,
                   style: str = 'colored',
                   preserve_transparency: bool = True,
                   min_contour_area: int = 10,
                   **kwargs) -> ProcessingResults:
    """
    Simplified function for image vectorization.
    
    Args:
        input_path: Path to input image
        output_path: Path to save SVG
        num_colors: Number of colors (None for auto mode)
        style: SVG style
        preserve_transparency: Whether to preserve transparency
        min_contour_area: Minimum contour area for processing (in pixels)
        **kwargs: Additional parameters
        
    Returns:
        Processing results
    """
    # Check if num_colors is provided to determine mode
    if num_colors is not None:
        # Manual mode
        vectorizer = ColorImageVectorizer(
            preserve_transparency=preserve_transparency,
            color_mode=ColorMode.MANUAL,
            num_colors=num_colors,
            min_contour_area=min_contour_area,
            **kwargs
        )
    else:
        # Auto mode
        vectorizer = ColorImageVectorizer(
            preserve_transparency=preserve_transparency,
            color_mode=ColorMode.AUTO,
            min_contour_area=min_contour_area,
            **kwargs
        )
    
    results = vectorizer.process_image(input_path, num_colors=num_colors)
    vectorizer.create_svg(results, output_path, style=style, preserve_transparency=preserve_transparency)
    return results


def create_vectorizer(auto_mode: bool = True, 
                     num_colors: Optional[int] = None,
                     min_contour_area: int = 10,
                     **kwargs) -> ColorImageVectorizer:
    """
    Create a vectorizer with specified mode.
    
    Args:
        auto_mode: Whether to use auto mode (True) or manual mode (False)
        num_colors: Number of colors for manual mode (required if auto_mode=False)
        min_contour_area: Minimum contour area for processing (in pixels)
        **kwargs: Additional parameters
        
    Returns:
        ColorImageVectorizer instance
    """
    if auto_mode:
        return ColorImageVectorizer(
            color_mode=ColorMode.AUTO,
            min_contour_area=min_contour_area,
            **kwargs
        )
    else:
        if num_colors is None:
            raise ValueError("num_colors must be provided for manual mode")
        return ColorImageVectorizer(
            color_mode=ColorMode.MANUAL,
            num_colors=num_colors,
            min_contour_area=min_contour_area,
            **kwargs
        )
