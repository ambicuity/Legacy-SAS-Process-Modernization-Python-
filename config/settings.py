"""
Configuration settings for SAS Modernization Prototype
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, CONFIG_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data processing settings
DATA_PROCESSING = {
    "default_cleaning_rules": {
        "remove_duplicates": True,
        "strip_whitespace": True,
        "standardize_case": "upper",
        "handle_missing": "flag"
    },
    "supported_file_types": ["csv", "excel", "sas"],
    "max_file_size_mb": 500,
    "chunk_size": 10000
}

# Statistical analysis settings
STATISTICAL_ANALYSIS = {
    "default_alpha": 0.05,
    "confidence_level": 0.95,
    "default_statistics": ["count", "mean", "std", "min", "max", "median"],
    "correlation_methods": ["pearson", "spearman", "kendall"],
    "precision_decimals": 4
}

# Report generation settings
REPORT_GENERATION = {
    "default_output_format": "html",
    "excel_engine": "openpyxl",
    "html_template": "main",
    "include_timestamp": True,
    "max_table_rows": 10000
}

# Clinical trial specific settings
CLINICAL_TRIAL = {
    "required_demographics": ["PATIENT_ID", "AGE", "SEX", "TREATMENT_GROUP"],
    "age_groups": {
        "pediatric": (0, 17),
        "adult": (18, 64),
        "elderly": (65, 120)
    },
    "adverse_event_severities": ["MILD", "MODERATE", "SEVERE"],
    "visit_types": ["BASELINE", "FOLLOWUP", "END_OF_STUDY"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": LOGS_DIR / "sas_modernization.log",
    "max_file_size_mb": 10,
    "backup_count": 5
}

# Validation rules
VALIDATION_RULES = {
    "age_range": (0, 120),
    "required_fields": ["PATIENT_ID"],
    "data_quality_thresholds": {
        "missing_data_percent": 50,
        "duplicate_threshold": 10
    }
}