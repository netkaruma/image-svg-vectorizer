from setuptools import setup, find_packages

setup(
    name="image_svg_vectorizer",
    version="1.0.0",
    author="netkaruma",
    author_email="",
    description="Convert raster images to SVG with color simplification and transparency support",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
)