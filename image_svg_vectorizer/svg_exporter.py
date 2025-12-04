"""
Module for exporting results to SVG.
"""

import json
from pathlib import Path
from xml.dom import minidom
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import numpy as np
import cv2

from .utils import ProcessingResults


class SVGExporter:
    """Class for exporting processing results to SVG."""
    
    def __init__(self, color_processor):
        """
        Initialize SVG exporter.
        
        Args:
            color_processor: ColorProcessor object for contour conversion
        """
        self.color_processor = color_processor
    
    def create_svg(self, 
                  results: ProcessingResults,
                  output_path: Union[str, Path],
                  style: str = 'colored',
                  stroke_width: float = 0.5,
                  stroke_color: str = '#000000',
                  add_stroke: bool = False,
                  flatten_layers: bool = False,
                  preserve_transparency: bool = True) -> str:
        """
        Create SVG file from processing results.
        
        Args:
            results: Image processing results
            output_path: Path to save SVG
            style: SVG style ('colored', 'outline', 'minimal', 'transparent')
            stroke_width: Stroke width
            stroke_color: Stroke color
            add_stroke: Whether to add stroke
            flatten_layers: Whether to merge all layers into one
            preserve_transparency: Whether to preserve transparency
            
        Returns:
            Path to saved file
        """
        height, width = results.image_shape[:2]
        
        # Create SVG document
        svg_doc = minidom.Document()
        svg_elem = svg_doc.createElement('svg')
        
        # SVG attributes
        svg_elem.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        svg_elem.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
        svg_elem.setAttribute('width', str(width))
        svg_elem.setAttribute('height', str(height))
        svg_elem.setAttribute('viewBox', f'0 0 {width} {height}')
        svg_elem.setAttribute('version', '1.1')
        
        # Add title with metadata
        title = svg_doc.createElement('title')
        title.appendChild(svg_doc.createTextNode(f'Vectorized image: {output_path}'))
        svg_elem.appendChild(title)
        
        desc = svg_doc.createElement('desc')
        desc.appendChild(svg_doc.createTextNode(
            f'Created by ColorImageVectorizer. Colors: {len(results.color_info)}, '
            f'Transparency: {results.transparency_info["has_alpha"] if results.transparency_info else False}'
        ))
        svg_elem.appendChild(desc)
        
        svg_doc.appendChild(svg_elem)
        
        # Create style based on selected option
        if style == 'outline':
            self._create_outline_svg(svg_doc, svg_elem, results, stroke_width, stroke_color)
        elif style == 'minimal':
            self._create_minimal_svg(svg_doc, svg_elem, results, flatten_layers)
        elif style == 'transparent' and preserve_transparency and results.transparency_info:
            self._create_transparent_svg(svg_doc, svg_elem, results, add_stroke, stroke_width, stroke_color)
        else:  # 'colored' by default
            self._create_colored_svg(svg_doc, svg_elem, results, add_stroke, stroke_width, stroke_color, preserve_transparency)
        
        # Save SVG
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_doc.toprettyxml(indent="  "))
        
        print(f"SVG saved: {output_path}")
        print(f"Size: {width}x{height}, Colors: {len(results.color_info)}, "
              f"With transparency: {preserve_transparency and results.transparency_info and results.transparency_info['has_alpha']}")
        
        return str(output_path)
    
    def _create_colored_svg(self, 
                           svg_doc: minidom.Document,
                           svg_elem: minidom.Element,
                           results: ProcessingResults,
                           add_stroke: bool,
                           stroke_width: float,
                           stroke_color: str,
                           preserve_transparency: bool) -> None:
        """Create colored SVG with fill and transparency."""
        
        # Group for fills
        fill_group = svg_doc.createElement('g')
        fill_group.setAttribute('id', 'color-fills')
        svg_elem.appendChild(fill_group)
        
        # Add each color
        for color_hex, color_data in results.color_info.items():
            # Get paths for this color
            external_paths = self.color_processor.contours_to_svg_path(color_data.external_contours)
            internal_paths = self.color_processor.contours_to_svg_path(color_data.internal_contours)
            
            if not external_paths and not internal_paths:
                continue
            
            # Create group for this color
            color_group = svg_doc.createElement('g')
            
            # Form color with transparency if needed
            if preserve_transparency and hasattr(color_data, 'alpha') and color_data.alpha < 255:
                # Add alpha channel to HEX
                fill_color = color_hex + f"{color_data.alpha:02x}"
                opacity = color_data.alpha / 255.0
                color_group.setAttribute('opacity', str(opacity))
                color_group.setAttribute('fill', color_hex)  # Main color without alpha
            else:
                fill_color = color_hex
                color_group.setAttribute('fill', fill_color)
            
            if add_stroke:
                color_group.setAttribute('stroke', stroke_color)
                color_group.setAttribute('stroke-width', str(stroke_width))
                color_group.setAttribute('stroke-linejoin', 'round')
            else:
                color_group.setAttribute('stroke', 'none')
            
            color_group.setAttribute('fill-rule', 'evenodd')
            color_group.setAttribute('data-color', color_hex)
            color_group.setAttribute('data-area', str(color_data.area))
            if hasattr(color_data, 'alpha'):
                color_group.setAttribute('data-alpha', str(color_data.alpha))
            
            # Process contours considering transparent areas inside
            if hasattr(color_data, 'transparent_areas') and color_data.transparent_areas is not None:
                # Create separate paths for opaque and partially transparent areas
                self._add_transparent_paths(svg_doc, color_group, color_data, fill_color)
            else:
                # Combine all paths of this color
                combined_d = ""
                for path in external_paths:
                    combined_d += path + " "
                for path in internal_paths:
                    combined_d += path + " "
                
                if combined_d.strip():
                    path_elem = svg_doc.createElement('path')
                    path_elem.setAttribute('d', combined_d.strip())
                    color_group.appendChild(path_elem)
            
            fill_group.appendChild(color_group)
    
    def _add_transparent_paths(self,
                              svg_doc: minidom.Document,
                              color_group: minidom.Element,
                              color_data,
                              fill_color: str) -> None:
        """Add paths with different transparency."""
        # Here you can implement more complex logic for partially transparent areas
        # For example, creating separate paths with different opacity
        
        # In the simplest case, just add all contours
        external_paths = self.color_processor.contours_to_svg_path(color_data.external_contours)
        internal_paths = self.color_processor.contours_to_svg_path(color_data.internal_contours)
        
        combined_d = ""
        for path in external_paths:
            combined_d += path + " "
        for path in internal_paths:
            combined_d += path + " "
        
        if combined_d.strip():
            path_elem = svg_doc.createElement('path')
            path_elem.setAttribute('d', combined_d.strip())
            color_group.appendChild(path_elem)
    
    def _create_transparent_svg(self,
                            svg_doc: minidom.Document,
                            svg_elem: minidom.Element,
                            results: ProcessingResults,
                            add_stroke: bool,
                            stroke_width: float,
                            stroke_color: str) -> None:
        """Create SVG with preserved transparency."""
        
        if not results.transparency_info or not results.transparency_info['has_alpha']:
            # If no transparency, use regular colored SVG
            self._create_colored_svg(svg_doc, svg_elem, results, add_stroke, stroke_width, stroke_color, False)
            return
        
        # Create group
        main_group = svg_doc.createElement('g')
        main_group.setAttribute('id', 'transparent-image')
        svg_elem.appendChild(main_group)
        
        # IMPORTANT: Do not add white background! Transparent areas should remain transparent
        # Instead, first add all opaque elements,
        # and transparent areas will remain empty
        
        # Add each color
        for color_hex, color_data in results.color_info.items():
            # Skip fully transparent "colors" (HEX ending with '00')
            if color_hex.endswith('00'):
                continue
                
            external_paths = self.color_processor.contours_to_svg_path(color_data.external_contours)
            internal_paths = self.color_processor.contours_to_svg_path(color_data.internal_contours)
            
            if not external_paths:
                continue
            
            # Create path
            combined_d = " ".join(external_paths + internal_paths)
            
            if combined_d.strip():
                path_elem = svg_doc.createElement('path')
                path_elem.setAttribute('d', combined_d.strip())
                
                # Use HEX with alpha channel if available
                if hasattr(color_data, 'alpha') and color_data.alpha < 255:
                    # HEX already contains alpha channel
                    path_elem.setAttribute('fill', color_hex)
                else:
                    # Without alpha channel
                    path_elem.setAttribute('fill', color_hex)
                    if hasattr(color_data, 'alpha'):
                        opacity = color_data.alpha / 255.0
                        path_elem.setAttribute('fill-opacity', str(opacity))
                
                if add_stroke:
                    path_elem.setAttribute('stroke', stroke_color)
                    path_elem.setAttribute('stroke-width', str(stroke_width))
                    path_elem.setAttribute('stroke-linejoin', 'round')
                    if hasattr(color_data, 'alpha'):
                        stroke_opacity = min(1.0, (color_data.alpha + 50) / 255.0)
                        path_elem.setAttribute('stroke-opacity', str(stroke_opacity))
                
                path_elem.setAttribute('fill-rule', 'evenodd')
                main_group.appendChild(path_elem)
    
    def _create_outline_svg(self,
                           svg_doc: minidom.Document,
                           svg_elem: minidom.Element,
                           results: ProcessingResults,
                           stroke_width: float,
                           stroke_color: str) -> None:
        """Create SVG with outlines only."""
        
        stroke_group = svg_doc.createElement('g')
        stroke_group.setAttribute('id', 'outlines')
        stroke_group.setAttribute('fill', 'none')
        stroke_group.setAttribute('stroke', stroke_color)
        stroke_group.setAttribute('stroke-width', str(stroke_width))
        stroke_group.setAttribute('stroke-linejoin', 'round')
        stroke_group.setAttribute('stroke-linecap', 'round')
        svg_elem.appendChild(stroke_group)
        
        # Combine all contours
        all_paths = []
        for color_data in results.color_info.values():
            all_paths.extend(self.color_processor.contours_to_svg_path(color_data.external_contours))
            all_paths.extend(self.color_processor.contours_to_svg_path(color_data.internal_contours))
        
        # Create single path for all contours (optimization)
        combined_d = " ".join(all_paths)
        
        if combined_d.strip():
            path_elem = svg_doc.createElement('path')
            path_elem.setAttribute('d', combined_d.strip())
            stroke_group.appendChild(path_elem)
    
    def _create_minimal_svg(self,
                           svg_doc: minidom.Document,
                           svg_elem: minidom.Element,
                           results: ProcessingResults,
                           flatten_layers: bool) -> None:
        """Create minimalistic SVG."""
        
        if flatten_layers:
            # Everything in one element
            combined_d = ""
            for color_hex, color_data in results.color_info.items():
                paths = self.color_processor.contours_to_svg_path(color_data.external_contours)
                paths.extend(self.color_processor.contours_to_svg_path(color_data.internal_contours))
                combined_d += " ".join(paths) + " "
            
            if combined_d.strip():
                path_elem = svg_doc.createElement('path')
                path_elem.setAttribute('d', combined_d.strip())
                path_elem.setAttribute('fill', '#000000')
                path_elem.setAttribute('fill-rule', 'evenodd')
                svg_elem.appendChild(path_elem)
        else:
            # Separate elements for each color
            for color_hex, color_data in results.color_info.items():
                paths = self.color_processor.contours_to_svg_path(color_data.external_contours)
                paths.extend(self.color_processor.contours_to_svg_path(color_data.internal_contours))
                
                if paths:
                    path_elem = svg_doc.createElement('path')
                    path_elem.setAttribute('d', " ".join(paths))
                    path_elem.setAttribute('fill', color_hex)
                    path_elem.setAttribute('fill-rule', 'evenodd')
                    
                    # Add transparency if available
                    if hasattr(color_data, 'alpha') and color_data.alpha < 255:
                        path_elem.setAttribute('fill-opacity', str(color_data.alpha / 255.0))
                    
                    svg_elem.appendChild(path_elem)
    
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
        stats = {
            "processing_time": results.processing_time,
            "image_size": {
                "width": results.image_shape[1],
                "height": results.image_shape[0]
            },
            "colors": {
                "total": len(results.unique_colors),
                "used": len(results.color_info),
                "details": []
            },
            "contours": {
                "total": 0,
                "external": 0,
                "internal": 0,
                "total_perimeter": 0.0
            },
            "transparency": {
                "has_transparency": results.transparency_info["has_alpha"] if results.transparency_info else False,
                "transparent_colors": 0,
                "fully_transparent_area": 0
            }
        }
        
        # Transparency statistics
        if results.transparency_info and results.transparency_info['has_alpha']:
            stats["transparency"]["fully_transparent_area"] = np.sum(
                results.transparency_info['fully_transparent']
            )
        
        total_area = results.image_shape[0] * results.image_shape[1]
        
        # Color statistics
        for color_hex, color_data in results.color_info.items():
            color_stats = {
                "hex": color_hex,
                "rgb": color_data.rgb,
                "area": color_data.area,
                "area_percentage": (color_data.area / total_area) * 100,
                "contours": {
                    "total": len(color_data.contours),
                    "external": len(color_data.external_contours),
                    "internal": len(color_data.internal_contours),
                    "external_perimeter": sum(color_data.external_perimeters),
                    "internal_perimeter": sum(color_data.internal_perimeters)
                }
            }
            
            # Add transparency information
            if hasattr(color_data, 'alpha'):
                color_stats["alpha"] = color_data.alpha
                color_stats["opacity"] = color_data.alpha / 255.0
                if color_data.alpha < 255:
                    stats["transparency"]["transparent_colors"] += 1
            
            stats["colors"]["details"].append(color_stats)
            
            # General contour statistics
            stats["contours"]["total"] += len(color_data.contours)
            stats["contours"]["external"] += len(color_data.external_contours)
            stats["contours"]["internal"] += len(color_data.internal_contours)
            stats["contours"]["total_perimeter"] += (
                sum(color_data.external_perimeters) + 
                sum(color_data.internal_perimeters)
            )
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Statistics saved: {output_path}")
        
        return stats


class SVGUtils:
    """Additional utilities for working with SVG."""
    
    @staticmethod
    def add_gradient(svg_content: str, 
                    gradient_id: str = 'gradient',
                    colors: List[str] = None,
                    opacity: float = 1.0) -> str:
        """
        Add gradient to SVG.
        
        Args:
            svg_content: SVG content
            gradient_id: Gradient ID
            colors: List of colors for gradient
            opacity: Gradient opacity
            
        Returns:
            Modified SVG
        """
        if colors is None:
            colors = ['#FF0000', '#00FF00', '#0000FF']
        
        # Parse SVG
        svg_doc = minidom.parseString(svg_content)
        svg_elem = svg_doc.documentElement
        
        # Create gradient element
        defs = svg_doc.createElement('defs')
        gradient = svg_doc.createElement('linearGradient')
        gradient.setAttribute('id', gradient_id)
        gradient.setAttribute('x1', '0%')
        gradient.setAttribute('y1', '0%')
        gradient.setAttribute('x2', '100%')
        gradient.setAttribute('y2', '100%')
        
        # Add gradient stops
        for i, color in enumerate(colors):
            stop = svg_doc.createElement('stop')
            offset = i * 100 // (len(colors) - 1) if len(colors) > 1 else 0
            stop.setAttribute('offset', f'{offset}%')
            stop.setAttribute('stop-color', color)
            if opacity < 1.0:
                stop.setAttribute('stop-opacity', str(opacity))
            gradient.appendChild(stop)
        
        defs.appendChild(gradient)
        
        # Insert defs at the beginning
        if svg_elem.firstChild:
            svg_elem.insertBefore(defs, svg_elem.firstChild)
        else:
            svg_elem.appendChild(defs)
        
        return svg_doc.toprettyxml()
    
    @staticmethod
    def optimize_svg(svg_content: str, 
                    precision: int = 2) -> str:
        """
        Optimize SVG by reducing coordinate precision.
        
        Args:
            svg_content: SVG content
            precision: Number of decimal places
            
        Returns:
            Optimized SVG
        """
        import re
        
        # Reduce number precision
        pattern = r'(\d+\.\d{' + str(precision + 1) + r',})(?=[^\.\d]|$)'
        
        def round_match(match):
            num = float(match.group(1))
            return f"{num:.{precision}f}"
        
        optimized = re.sub(pattern, round_match, svg_content)
        
        return optimized
    
    @staticmethod
    def add_transparency_filter(svg_content: str,
                               filter_id: str = 'transparency') -> str:
        """
        Add transparency filter to SVG.
        
        Args:
            svg_content: SVG content
            filter_id: Filter ID
            
        Returns:
            Modified SVG
        """
        svg_doc = minidom.parseString(svg_content)
        svg_elem = svg_doc.documentElement
        
        # Create filter element
        defs = svg_doc.createElement('defs')
        filter_elem = svg_doc.createElement('filter')
        filter_elem.setAttribute('id', filter_id)
        
        # Add transparency effect
        fe_component_transfer = svg_doc.createElement('feComponentTransfer')
        fe_component_transfer.setAttribute('result', 'transparency')
        
        fe_func_a = svg_doc.createElement('feFuncA')
        fe_func_a.setAttribute('type', 'table')
        fe_func_a.setAttribute('tableValues', '0 0.5 1')
        
        fe_component_transfer.appendChild(fe_func_a)
        filter_elem.appendChild(fe_component_transfer)
        
        defs.appendChild(filter_elem)
        
        # Insert defs at the beginning
        if svg_elem.firstChild:
            svg_elem.insertBefore(defs, svg_elem.firstChild)
        else:
            svg_elem.appendChild(defs)
        
        return svg_doc.toprettyxml()