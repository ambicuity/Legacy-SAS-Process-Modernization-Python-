"""
SAS Modernization Prototype

A Python-based modernization of legacy SAS processes for clinical trial reporting.
This package provides automated reporting and data analysis capabilities with
rigorous testing to ensure accuracy and reliability.
"""

__version__ = "1.0.0"
__author__ = "Clinical Trial Reporting Team"

from .data_processor import DataProcessor
from .statistical_analyzer import StatisticalAnalyzer
from .report_generator import ReportGenerator

__all__ = ["DataProcessor", "StatisticalAnalyzer", "ReportGenerator"]