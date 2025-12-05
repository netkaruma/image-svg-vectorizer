# Color Image Vectorizer
### A powerful Python tool for converting raster images (PNG, JPG, etc.) into scalable vector graphics (SVG) with optimized color simplification and transparency support.
___ 
<img width="1134" height="671" alt="image" src="https://github.com/user-attachments/assets/09b40db0-fcf0-4d75-9197-b4c684a4c1eb" />   

## Quick Start

## Installation

```bash
pip install git+https://github.com/netkaruma/image-svg-vectorizer.git
```

## Basic Usage

```python
from color_image_vectorizer import vectorize_image

# Single image vectorization
result = vectorize_image(
    input_path="input.png",
    output_path="output.svg",
    num_colors=8,
    preserve_transparency=True
)

# Or use the class directly
from color_image_vectorizer import ColorImageVectorizer

vectorizer = ColorImageVectorizer(
    preserve_transparency=True,
    use_acceleration=True  # Enable Numba acceleration
)

# Process image
results = vectorizer.process_image(
    "input.png",
    num_colors=8,
    method='kmeans',  # or 'median'
    auto_colors=False
)

# Create SVG
vectorizer.create_svg(
    results,
    "output.svg",
    style='colored',  # or 'outline', 'minimal', 'transparent'
    add_stroke=False
)

# Create preview
vectorizer.create_preview(results, "preview.png")
```

## Batch Processing

```python
from color_image_vectorizer import batch_vectorize

# Process all images in a directory
results = batch_vectorize(
    input_dir="./images",
    output_dir="./vectors",
    num_colors=8,
    style='colored',
    preserve_transparency=True
)
```

___

# Detailed Documentation

## Class: ColorImageVectorizer

### Main class for vectorizing color images with acceleration support.

### Initialization Parameters

```python
ColorImageVectorizer(
    simplify_contours: bool = True,
    contour_tolerance: float = 1.5,
    min_contour_area: int = 10,
    preserve_transparency: bool = True,
    use_acceleration: bool = True
)
```
### Parameters:

1. simplify_contours (bool): Whether to simplify contours using Douglas-Peucker algorithm. Reduces file size but may lose detail.
2. contour_tolerance (float): Tolerance for contour simplification. Higher values = more simplification.
3. min_contour_area (int): Minimum contour area (in pixels) to include in SVG. Smaller contours are discarded.
4. preserve_transparency (bool): Whether to preserve alpha channel from PNG images.
5. use_acceleration (bool): Use Numba JIT compiler for faster processing (requires numba package).

## Method: process_image()

### Process an image and extract color information.

```python
process_image(
    image_path: Union[str, Path],
    num_colors: int = 8,
    method: str = 'kmeans',
    auto_colors: bool = False,
    max_colors: int = 16
) -> ProcessingResults
```

## Parameters:

1. image_path (str/Path): Path to input image or numpy array. Supports PNG, JPG, BMP, TIFF, WebP.
2. num_colors (int): Target number of colors for simplification.
3. method (str): Color simplification method:
    'kmeans': K-means clustering (more accurate, slower)
    'median': Median cut quantization (faster, good for images with clear color separation)
4. auto_colors (bool): Automatically determine optimal number of colors based on image complexity.
5. max_colors (int): Maximum number of colors when auto_colors=True.

## Returns: ProcessingResults object containing:

1. Original and simplified images
2. Color information dictionaries
3. Contour data
4. Transparency information
5. Processing statistics

## Method: create_svg()

### Create SVG file from processing results.

```python
create_svg(
    results: ProcessingResults,
    output_path: Union[str, Path],
    style: str = 'colored',
    stroke_width: float = 0.5,
    stroke_color: str = '#000000',
    add_stroke: bool = False,
    flatten_layers: bool = False,
    preserve_transparency: bool = None
) -> str
```

## Parameters:
1. results (ProcessingResults): Results from process_image().
2. output_path (str/Path): Path to save SVG file.
3. style (str): SVG output style:
    1. 'colored': Full color fills (default)
    2. 'outline': Only outlines/contours
    3. 'minimal': Minimal SVG with reduced elements
    4. 'transparent': Preserve transparency (for PNG with alpha)
4. stroke_width (float): Width of stroke in SVG units.
5. stroke_color (str): Color of stroke in HEX format.
6. add_stroke (bool): Whether to add stroke outline to shapes.
7. flatten_layers (bool): Merge all color layers into single elements.
8. preserve_transparency (bool): Override transparency preservation (defaults to class setting).

## Method: create_preview()

### Generate comparison preview image.

```python
create_preview(
    results: ProcessingResults,
    output_path: Union[str, Path] = None,
    show: bool = True,
    show_alpha: bool = True
) -> Optional[np.ndarray]
```

## Parameters:

1. results (ProcessingResults): Processing results.
2. output_path (str/Path): Optional path to save preview PNG.
3. show (bool): Display preview using matplotlib.
4. show_alpha (bool): Show transparency information in preview.

## Method: export_stats()

### Export processing statistics to JSON.

```python
export_stats(
    results: ProcessingResults,
    output_path: Union[str, Path]
) -> Dict[str, Any]
```

## Parameters:
1. results (ProcessingResults): Processing results.
2. output_path (str/Path): Path to save JSON file.

## Function: vectorize_image()

### Convenience function for quick vectorization.

```python
vectorize_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    num_colors: int = 8,
    style: str = 'colored',
    preserve_transparency: bool = True,
    **kwargs
) -> ProcessingResults
```

### Additional kwargs: All parameters from ColorImageVectorizer and process_image().

___

## Class: BatchProcessor

### Process multiple images in batch mode.

```python
from color_image_vectorizer import BatchProcessor

processor = BatchProcessor(vectorizer=None)  # Optional custom vectorizer

processor.process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    num_colors: int = 8,
    style: str = 'colored',
    **kwargs
) -> List[ProcessingResults]
```

## Function: batch_vectorize()

### Convenience function for batch processing.

```python
batch_vectorize(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    num_colors: int = 8,
    style: str = 'colored',
    **kwargs
) -> List[ProcessingResults]
```
___

## Advanced Features

## Transparency Support

## The tool fully supports PNG transparency:

1. Preserves alpha channel in SVG output
2. Handles partially transparent pixels
3. Can separate fully transparent areas

```python
# Process with transparency
results = vectorizer.process_image(
    "image_with_alpha.png",
    preserve_transparency=True
)

# SVG will maintain transparent areas
vectorizer.create_svg(
    results,
    "output.svg",
    style='transparent'  # Or 'colored' with transparency
)
```

## Acceleration with Numba

### For large images, enable Numba acceleration:

```bash
pip install numba
```

```python
vectorizer = ColorImageVectorizer(use_acceleration=True)
```

## Automatic Color Detection

## Let the tool determine optimal color count:

```python
results = vectorizer.process_image(
    "input.jpg",
    auto_colors=True,
    max_colors=12  # Maximum colors to use
)
```

## Custom Contour Processing

```python
vectorizer = ColorImageVectorizer(
    simplify_contours=True,
    contour_tolerance=2.0,  # Higher = more simplification
    min_contour_area=20     # Ignore small details
)
```
