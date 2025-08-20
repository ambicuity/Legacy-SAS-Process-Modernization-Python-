"""
Data Processor Module

Replaces legacy SAS DATA steps with modern Python pandas operations.
Handles data loading, cleaning, transformation, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Modernizes SAS DATA step functionality using pandas.
    
    This class provides methods to:
    - Load clinical trial data from various formats
    - Clean and validate data according to clinical standards
    - Transform data for analysis and reporting
    - Apply business rules and derivations
    """
    
    def __init__(self):
        self.datasets = {}
        self.metadata = {}
        
    def load_data(self, file_path: str, dataset_name: str, 
                  file_type: str = 'csv') -> pd.DataFrame:
        """
        Load clinical trial data from file.
        
        Replaces SAS: PROC IMPORT or DATA step with INFILE
        
        Args:
            file_path: Path to data file
            dataset_name: Name to assign to the dataset
            file_type: File format ('csv', 'excel', 'sas')
            
        Returns:
            Loaded DataFrame
        """
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_type.lower() == 'excel':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Store dataset
            self.datasets[dataset_name] = df
            
            # Generate metadata
            self.metadata[dataset_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_counts': df.isnull().sum().to_dict()
            }
            
            logger.info(f"Loaded dataset '{dataset_name}': {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def clean_data(self, dataset_name: str, 
                   rules: Optional[Dict] = None) -> pd.DataFrame:
        """
        Clean and validate clinical trial data.
        
        Replaces SAS data cleaning steps with validation rules.
        
        Args:
            dataset_name: Name of dataset to clean
            rules: Dictionary of cleaning rules
            
        Returns:
            Cleaned DataFrame
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
            
        df = self.datasets[dataset_name].copy()
        
        # Default cleaning rules
        default_rules = {
            'remove_duplicates': True,
            'strip_whitespace': True,
            'standardize_case': 'upper',
            'handle_missing': 'flag'
        }
        
        if rules:
            default_rules.update(rules)
            
        # Apply cleaning rules
        if default_rules.get('remove_duplicates'):
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                logger.warning(f"Removed {removed_rows} duplicate rows from '{dataset_name}'")
        
        if default_rules.get('strip_whitespace'):
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                df[col] = df[col].astype(str).str.strip()
        
        if default_rules.get('standardize_case') == 'upper':
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                df[col] = df[col].astype(str).str.upper()
        
        # Update stored dataset
        self.datasets[dataset_name] = df
        
        return df
    
    def derive_variables(self, dataset_name: str, 
                        derivations: Dict[str, str]) -> pd.DataFrame:
        """
        Create derived variables from existing data.
        
        Replaces SAS variable derivation logic in DATA steps.
        
        Args:
            dataset_name: Name of dataset
            derivations: Dictionary mapping new variable names to derivation logic
            
        Returns:
            DataFrame with derived variables
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
            
        df = self.datasets[dataset_name].copy()
        
        for var_name, derivation in derivations.items():
            try:
                # Evaluate derivation (simplified example)
                df[var_name] = eval(derivation, {"df": df, "np": np, "pd": pd})
                logger.info(f"Created derived variable '{var_name}' in dataset '{dataset_name}'")
            except Exception as e:
                logger.error(f"Error creating derived variable '{var_name}': {str(e)}")
                raise
        
        # Update stored dataset
        self.datasets[dataset_name] = df
        
        return df
    
    def merge_datasets(self, left_dataset: str, right_dataset: str,
                      merge_keys: Union[str, List[str]], 
                      how: str = 'inner',
                      result_name: Optional[str] = None) -> pd.DataFrame:
        """
        Merge two datasets based on common keys.
        
        Replaces SAS MERGE statement in DATA step.
        
        Args:
            left_dataset: Name of left dataset
            right_dataset: Name of right dataset
            merge_keys: Column name(s) to merge on
            how: Type of merge ('inner', 'outer', 'left', 'right')
            result_name: Name for merged dataset
            
        Returns:
            Merged DataFrame
        """
        if left_dataset not in self.datasets:
            raise ValueError(f"Dataset '{left_dataset}' not found")
        if right_dataset not in self.datasets:
            raise ValueError(f"Dataset '{right_dataset}' not found")
            
        left_df = self.datasets[left_dataset]
        right_df = self.datasets[right_dataset]
        
        merged_df = pd.merge(left_df, right_df, on=merge_keys, how=how)
        
        if result_name:
            self.datasets[result_name] = merged_df
            logger.info(f"Created merged dataset '{result_name}': {len(merged_df)} rows")
        
        return merged_df
    
    def filter_data(self, dataset_name: str, condition: str,
                   result_name: Optional[str] = None) -> pd.DataFrame:
        """
        Filter dataset based on condition.
        
        Replaces SAS WHERE statement or subsetting IF.
        
        Args:
            dataset_name: Name of dataset to filter
            condition: Boolean condition for filtering
            result_name: Name for filtered dataset
            
        Returns:
            Filtered DataFrame
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
            
        df = self.datasets[dataset_name]
        
        try:
            # Evaluate condition
            filtered_df = df.query(condition)
            
            if result_name:
                self.datasets[result_name] = filtered_df
                logger.info(f"Created filtered dataset '{result_name}': {len(filtered_df)} rows")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering dataset '{dataset_name}': {str(e)}")
            raise
    
    def get_dataset_summary(self, dataset_name: str) -> Dict:
        """
        Get summary information about a dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with dataset summary
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
            
        df = self.datasets[dataset_name]
        
        return {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }