"""
IVI Water Trends - Interactive Visualization and Impact Assessment Tool
for Seasonal Surface Water Trends using CoRE Stack data.
"""

__version__ = "0.1.0"
__author__ = "IVI Water Trends Team"
__email__ = "contact@ivi-water.org"

from .api_client import CoREStackClient
from .data_processor import DataProcessor
from .visualizer import WaterTrendsVisualizer
from .export_utils import ExportUtils

__all__ = [
    "CoREStackClient",
    "DataProcessor", 
    "WaterTrendsVisualizer",
    "ExportUtils"
]
