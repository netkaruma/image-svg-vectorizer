"""
Module for batch image processing.
"""

from pathlib import Path
from typing import List, Union
from .vectorizer import ColorImageVectorizer
from .utils import ProcessingResults


class BatchProcessor:
    """Class for batch image processing."""
    
    def __init__(self, vectorizer: ColorImageVectorizer = None):
        """
        Initialize batch processor.
        
        Args:
            vectorizer: Vectorizer object (if None, creates new one)
        """
        self.vectorizer = vectorizer or ColorImageVectorizer()
    
    def process_directory(self,
                         input_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         num_colors: int = 8,
                         style: str = 'colored',
                         **kwargs) -> List[ProcessingResults]:
        """
        Batch processing of images in a directory.
        
        Args:
            input_dir: Input directory with images
            output_dir: Output directory for SVG
            num_colors: Number of colors
            style: SVG style
            **kwargs: Additional parameters
            
        Returns:
            List of processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_list = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        image_files = []
        for ext in supported_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images for processing")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {image_file.name}")
            
            try:
                # Process image
                results = self.vectorizer.process_image(
                    image_file,
                    num_colors=num_colors,
                    **{k: v for k, v in kwargs.items() if k in ['method', 'auto_colors', 'max_colors']}
                )
                
                # Create SVG
                svg_filename = image_file.stem + '.svg'
                svg_path = output_dir / svg_filename
                
                self.vectorizer.create_svg(
                    results,
                    svg_path,
                    style=style,
                    **{k: v for k, v in kwargs.items() if k in [
                        'stroke_width', 'stroke_color', 'add_stroke', 'flatten_layers'
                    ]}
                )
                
                # Save preview
                preview_path = output_dir / f"{image_file.stem}_preview.png"
                self.vectorizer.create_preview(results, preview_path, show=False)
                
                # Save statistics
                stats_path = output_dir / f"{image_file.stem}_stats.json"
                self.vectorizer.export_stats(results, stats_path)
                
                results_list.append(results)
                
                print(f"  ✓ Successfully processed")
                
            except Exception as e:
                print(f"  ✗ Error processing {image_file.name}: {str(e)}")
        
        print(f"\nBatch processing completed. Processed: {len(results_list)}/{len(image_files)}")
        
        return results_list


def batch_vectorize(input_dir: Union[str, Path],
                   output_dir: Union[str, Path],
                   num_colors: int = 8,
                   style: str = 'colored',
                   **kwargs) -> List[ProcessingResults]:
    """
    Batch vectorization of images in a directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        num_colors: Number of colors
        style: SVG style
        **kwargs: Additional parameters
        
    Returns:
        List of results
    """
    processor = BatchProcessor()
    return processor.process_directory(input_dir, output_dir, num_colors, style, **kwargs)