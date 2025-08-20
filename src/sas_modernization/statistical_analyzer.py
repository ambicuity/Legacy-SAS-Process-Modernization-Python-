"""
Statistical Analyzer Module

Replaces legacy SAS PROC steps with modern Python statistical analysis.
Provides comprehensive statistical analysis capabilities for clinical trial data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Modernizes SAS PROC functionality using pandas, numpy, and scipy.
    
    This class provides methods to:
    - Generate descriptive statistics (PROC MEANS, PROC UNIVARIATE)
    - Create frequency tables (PROC FREQ)
    - Perform statistical tests (PROC TTEST, PROC ANOVA)
    - Generate correlation analysis (PROC CORR)
    - Create summary tables for clinical reporting
    """
    
    def __init__(self):
        self.results = {}
        
    def proc_means(self, data: pd.DataFrame, 
                   variables: Optional[List[str]] = None,
                   by_variables: Optional[List[str]] = None,
                   statistics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate descriptive statistics.
        
        Replaces SAS: PROC MEANS
        
        Args:
            data: Input DataFrame
            variables: List of variables to analyze (None for all numeric)
            by_variables: Variables to group by
            statistics: List of statistics to compute
            
        Returns:
            DataFrame with descriptive statistics
        """
        if statistics is None:
            statistics = ['count', 'mean', 'std', 'min', 'max']
        
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()
        
        try:
            if by_variables:
                # Group by analysis
                grouped = data.groupby(by_variables)
                results = []
                
                for name, group in grouped:
                    group_stats = group[variables].describe().T
                    group_stats['group'] = str(name) if not isinstance(name, tuple) else '_'.join(map(str, name))
                    results.append(group_stats)
                
                result_df = pd.concat(results)
            else:
                # Overall analysis
                result_df = data[variables].describe().T
            
            # Store results
            self.results['proc_means'] = result_df
            logger.info(f"PROC MEANS completed for {len(variables)} variables")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in PROC MEANS: {str(e)}")
            raise
    
    def proc_freq(self, data: pd.DataFrame,
                  variables: List[str],
                  by_variables: Optional[List[str]] = None,
                  include_percentages: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate frequency tables.
        
        Replaces SAS: PROC FREQ
        
        Args:
            data: Input DataFrame
            variables: Variables to analyze
            by_variables: Variables to group by
            include_percentages: Include percentage calculations
            
        Returns:
            Dictionary of frequency tables
        """
        try:
            freq_tables = {}
            
            for var in variables:
                if var not in data.columns:
                    logger.warning(f"Variable '{var}' not found in data")
                    continue
                
                if by_variables:
                    # Cross-tabulation
                    for by_var in by_variables:
                        table_name = f"{var}_by_{by_var}"
                        freq_table = pd.crosstab(data[var], data[by_var], 
                                               margins=True, 
                                               normalize='columns' if include_percentages else False)
                        freq_tables[table_name] = freq_table
                else:
                    # Simple frequency
                    freq_table = data[var].value_counts(dropna=False).to_frame('frequency')
                    
                    if include_percentages:
                        freq_table['percentage'] = (freq_table['frequency'] / 
                                                  freq_table['frequency'].sum() * 100)
                    
                    freq_tables[var] = freq_table
            
            # Store results
            self.results['proc_freq'] = freq_tables
            logger.info(f"PROC FREQ completed for {len(variables)} variables")
            
            return freq_tables
            
        except Exception as e:
            logger.error(f"Error in PROC FREQ: {str(e)}")
            raise
    
    def proc_ttest(self, data: pd.DataFrame,
                   test_variable: str,
                   group_variable: Optional[str] = None,
                   mu: float = 0,
                   alpha: float = 0.05) -> Dict[str, Union[float, str]]:
        """
        Perform t-test analysis.
        
        Replaces SAS: PROC TTEST
        
        Args:
            data: Input DataFrame
            test_variable: Variable to test
            group_variable: Grouping variable for two-sample test
            mu: Null hypothesis mean (for one-sample test)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        try:
            results = {}
            
            if group_variable is None:
                # One-sample t-test
                test_data = data[test_variable].dropna()
                t_stat, p_value = stats.ttest_1samp(test_data, mu)
                
                results.update({
                    'test_type': 'One-sample t-test',
                    'variable': test_variable,
                    'n': len(test_data),
                    'mean': test_data.mean(),
                    'std': test_data.std(),
                    'null_hypothesis_mean': mu,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'alpha': alpha
                })
                
            else:
                # Two-sample t-test
                unique_groups = data[group_variable].unique()
                
                if len(unique_groups) < 2:
                    raise ValueError("Group variable must have at least 2 levels for t-test")
                elif len(unique_groups) > 2:
                    # Use first two groups and warn
                    logger.warning(f"More than 2 groups found. Using first 2 groups: {unique_groups[:2]}")
                    unique_groups = unique_groups[:2]
                
                # Get data for each group
                group1_data = data[data[group_variable] == unique_groups[0]][test_variable].dropna()
                group2_data = data[data[group_variable] == unique_groups[1]][test_variable].dropna()
                
                # Perform equal variance test
                levene_stat, levene_p = stats.levene(group1_data, group2_data)
                equal_var = levene_p > alpha
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
                
                results.update({
                    'test_type': 'Two-sample t-test',
                    'variable': test_variable,
                    'group_variable': group_variable,
                    'group1': unique_groups[0],
                    'group2': unique_groups[1],
                    'n1': len(group1_data),
                    'n2': len(group2_data),
                    'mean1': group1_data.mean(),
                    'mean2': group2_data.mean(),
                    'std1': group1_data.std(),
                    'std2': group2_data.std(),
                    'equal_variance': equal_var,
                    'levene_p': levene_p,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'alpha': alpha
                })
            
            # Store results
            self.results['proc_ttest'] = results
            logger.info(f"PROC TTEST completed for variable '{test_variable}'")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in PROC TTEST: {str(e)}")
            raise
    
    def proc_corr(self, data: pd.DataFrame,
                  variables: Optional[List[str]] = None,
                  method: str = 'pearson') -> pd.DataFrame:
        """
        Generate correlation analysis.
        
        Replaces SAS: PROC CORR
        
        Args:
            data: Input DataFrame
            variables: Variables to correlate (None for all numeric)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        try:
            if variables is None:
                variables = data.select_dtypes(include=[np.number]).columns.tolist()
            
            corr_data = data[variables]
            
            # Calculate correlations
            if method == 'pearson':
                corr_matrix = corr_data.corr()
            elif method == 'spearman':
                corr_matrix = corr_data.corr(method='spearman')
            elif method == 'kendall':
                corr_matrix = corr_data.corr(method='kendall')
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
            
            # Calculate p-values
            n = len(corr_data)
            pval_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
            
            for i, var1 in enumerate(corr_matrix.index):
                for j, var2 in enumerate(corr_matrix.columns):
                    if i <= j:  # Only calculate upper triangle
                        if var1 == var2:
                            pval_matrix.loc[var1, var2] = 0
                        else:
                            if method == 'pearson':
                                _, p_val = stats.pearsonr(corr_data[var1].dropna(), 
                                                        corr_data[var2].dropna())
                            elif method == 'spearman':
                                _, p_val = stats.spearmanr(corr_data[var1].dropna(), 
                                                         corr_data[var2].dropna())
                            else:  # kendall
                                _, p_val = stats.kendalltau(corr_data[var1].dropna(), 
                                                          corr_data[var2].dropna())
                            
                            pval_matrix.loc[var1, var2] = p_val
                            pval_matrix.loc[var2, var1] = p_val
            
            # Store results
            self.results['proc_corr'] = {
                'correlation_matrix': corr_matrix,
                'p_value_matrix': pval_matrix,
                'method': method,
                'n_observations': n
            }
            
            logger.info(f"PROC CORR completed using {method} method for {len(variables)} variables")
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"Error in PROC CORR: {str(e)}")
            raise
    
    def create_summary_table(self, data: pd.DataFrame,
                           variables: List[str],
                           group_variable: Optional[str] = None,
                           include_tests: bool = True) -> pd.DataFrame:
        """
        Create comprehensive summary table for clinical reporting.
        
        Combines multiple PROC steps into a single summary table.
        
        Args:
            data: Input DataFrame
            variables: Variables to summarize
            group_variable: Grouping variable
            include_tests: Include statistical tests between groups
            
        Returns:
            Summary table DataFrame
        """
        try:
            summary_rows = []
            
            for var in variables:
                if var not in data.columns:
                    continue
                
                var_data = data[var].dropna()
                
                # Determine variable type
                if pd.api.types.is_numeric_dtype(var_data):
                    # Continuous variable
                    if group_variable:
                        for group_name in data[group_variable].unique():
                            group_data = data[data[group_variable] == group_name][var].dropna()
                            summary_rows.append({
                                'Variable': var,
                                'Group': group_name,
                                'N': len(group_data),
                                'Mean': group_data.mean(),
                                'Std': group_data.std(),
                                'Median': group_data.median(),
                                'Q1': group_data.quantile(0.25),
                                'Q3': group_data.quantile(0.75),
                                'Min': group_data.min(),
                                'Max': group_data.max()
                            })
                        
                        # Add statistical test
                        if include_tests and len(data[group_variable].unique()) == 2:
                            test_result = self.proc_ttest(data, var, group_variable)
                            summary_rows.append({
                                'Variable': f"{var} (p-value)",
                                'Group': 'Statistical Test',
                                'N': f"t={test_result['t_statistic']:.3f}",
                                'Mean': f"p={test_result['p_value']:.4f}",
                                'Std': 'Significant' if test_result['significant'] else 'Not Significant',
                                'Median': '', 'Q1': '', 'Q3': '', 'Min': '', 'Max': ''
                            })
                    else:
                        summary_rows.append({
                            'Variable': var,
                            'Group': 'Overall',
                            'N': len(var_data),
                            'Mean': var_data.mean(),
                            'Std': var_data.std(),
                            'Median': var_data.median(),
                            'Q1': var_data.quantile(0.25),
                            'Q3': var_data.quantile(0.75),
                            'Min': var_data.min(),
                            'Max': var_data.max()
                        })
                        
                else:
                    # Categorical variable
                    freq_table = self.proc_freq(data, [var])
                    
                    if group_variable:
                        cross_tab = pd.crosstab(data[var], data[group_variable], margins=True)
                        
                        for group_name in cross_tab.columns[:-1]:  # Exclude 'All' column
                            for category in cross_tab.index[:-1]:  # Exclude 'All' row
                                count = cross_tab.loc[category, group_name]
                                total = cross_tab.loc['All', group_name]
                                pct = (count / total * 100) if total > 0 else 0
                                
                                summary_rows.append({
                                    'Variable': f"{var} ({category})",
                                    'Group': group_name,
                                    'N': f"{count} ({pct:.1f}%)",
                                    'Mean': '', 'Std': '', 'Median': '', 
                                    'Q1': '', 'Q3': '', 'Min': '', 'Max': ''
                                })
                    else:
                        for category, count in freq_table[var]['frequency'].items():
                            total = freq_table[var]['frequency'].sum()
                            pct = (count / total * 100) if total > 0 else 0
                            
                            summary_rows.append({
                                'Variable': f"{var} ({category})",
                                'Group': 'Overall',
                                'N': f"{count} ({pct:.1f}%)",
                                'Mean': '', 'Std': '', 'Median': '',
                                'Q1': '', 'Q3': '', 'Min': '', 'Max': ''
                            })
            
            summary_df = pd.DataFrame(summary_rows)
            
            # Store results
            self.results['summary_table'] = summary_df
            logger.info(f"Summary table created for {len(variables)} variables")
            
            return summary_df
            
        except Exception as e:
            logger.error(f"Error creating summary table: {str(e)}")
            raise
    
    def get_all_results(self) -> Dict:
        """Get all stored analysis results."""
        return self.results.copy()
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()
        logger.info("All analysis results cleared")