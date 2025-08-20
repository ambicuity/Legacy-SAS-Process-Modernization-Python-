"""
SAS Modernization Prototype - Main Application

This module demonstrates the modernization of legacy SAS processes using Python.
It orchestrates the complete workflow from data processing to report generation.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from sas_modernization import DataProcessor, StatisticalAnalyzer, ReportGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sas_modernization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ClinicalTrialProcessor:
    """
    Main orchestrator for clinical trial data processing and reporting.
    
    This class demonstrates how the modernization prototype can replace
    entire SAS workflows with Python-based solutions.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator()
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_sample_data(self, n_patients: int = 500) -> Dict[str, pd.DataFrame]:
        """
        Generate sample clinical trial data for demonstration.
        
        In production, this would be replaced by data loading from actual sources.
        
        Args:
            n_patients: Number of patients to simulate
            
        Returns:
            Dictionary of sample datasets
        """
        logger.info(f"Generating sample clinical trial data for {n_patients} patients")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate demographics data
        demographics = pd.DataFrame({
            'PATIENT_ID': [f"PAT_{i:04d}" for i in range(1, n_patients + 1)],
            'AGE': np.random.normal(65, 12, n_patients).astype(int),
            'SEX': np.random.choice(['M', 'F'], n_patients, p=[0.45, 0.55]),
            'RACE': np.random.choice(['WHITE', 'BLACK', 'ASIAN', 'OTHER'], n_patients, p=[0.7, 0.15, 0.1, 0.05]),
            'TREATMENT_GROUP': np.random.choice(['TREATMENT', 'PLACEBO'], n_patients, p=[0.5, 0.5]),
            'STUDY_SITE': np.random.choice([f'SITE_{i:02d}' for i in range(1, 21)], n_patients),
            'ENROLLMENT_DATE': pd.date_range('2023-01-01', periods=n_patients, freq='D')[::-1]
        })
        
        # Ensure age is within reasonable bounds
        demographics['AGE'] = np.clip(demographics['AGE'], 18, 90)
        
        # Generate baseline measurements
        baseline_measurements = pd.DataFrame({
            'PATIENT_ID': demographics['PATIENT_ID'],
            'HEIGHT_CM': np.random.normal(170, 10, n_patients),
            'WEIGHT_KG': np.random.normal(75, 15, n_patients),
            'SYSTOLIC_BP': np.random.normal(130, 20, n_patients),
            'DIASTOLIC_BP': np.random.normal(80, 12, n_patients),
            'HEART_RATE': np.random.normal(72, 12, n_patients),
            'CHOLESTEROL': np.random.normal(200, 40, n_patients),
            'GLUCOSE': np.random.normal(100, 20, n_patients)
        })
        
        # Calculate BMI
        baseline_measurements['BMI'] = (
            baseline_measurements['WEIGHT_KG'] / 
            (baseline_measurements['HEIGHT_CM'] / 100) ** 2
        ).round(1)
        
        # Generate follow-up measurements (simulate treatment effect)
        followup_measurements = baseline_measurements.copy()
        followup_measurements['VISIT'] = 'FOLLOWUP'
        baseline_measurements['VISIT'] = 'BASELINE'
        
        # Simulate treatment effects
        treatment_patients = demographics[demographics['TREATMENT_GROUP'] == 'TREATMENT']['PATIENT_ID']
        
        for patient_id in treatment_patients:
            idx = followup_measurements['PATIENT_ID'] == patient_id
            # Simulate modest improvements in treatment group
            followup_measurements.loc[idx, 'SYSTOLIC_BP'] *= 0.95  # 5% reduction
            followup_measurements.loc[idx, 'DIASTOLIC_BP'] *= 0.93  # 7% reduction
            followup_measurements.loc[idx, 'CHOLESTEROL'] *= 0.92   # 8% reduction
            followup_measurements.loc[idx, 'GLUCOSE'] *= 0.96       # 4% reduction
        
        # Combine baseline and follow-up
        all_measurements = pd.concat([baseline_measurements, followup_measurements], ignore_index=True)
        
        # Generate adverse events data
        ae_rate = 0.3  # 30% of patients experience at least one AE
        ae_patients = np.random.choice(demographics['PATIENT_ID'], 
                                     int(n_patients * ae_rate), 
                                     replace=False)
        
        adverse_events = []
        ae_terms = ['HEADACHE', 'NAUSEA', 'FATIGUE', 'DIZZINESS', 'RASH', 'INSOMNIA']
        ae_severities = ['MILD', 'MODERATE', 'SEVERE']
        
        for patient_id in ae_patients:
            # Some patients may have multiple AEs
            n_aes = np.random.poisson(1.2) + 1
            for _ in range(n_aes):
                adverse_events.append({
                    'PATIENT_ID': patient_id,
                    'AE_TERM': np.random.choice(ae_terms),
                    'SEVERITY': np.random.choice(ae_severities, p=[0.6, 0.3, 0.1]),
                    'START_DATE': np.random.choice(pd.date_range('2023-02-01', '2023-08-01')),
                    'RELATED_TO_TREATMENT': np.random.choice(['YES', 'NO'], p=[0.7, 0.3])
                })
        
        adverse_events_df = pd.DataFrame(adverse_events)
        
        # Generate lab results
        lab_results = pd.DataFrame({
            'PATIENT_ID': np.repeat(demographics['PATIENT_ID'], 2),
            'VISIT': ['BASELINE'] * n_patients + ['FOLLOWUP'] * n_patients,
            'HEMOGLOBIN': np.random.normal(13.5, 2, n_patients * 2),
            'WHITE_BLOOD_CELLS': np.random.normal(6.5, 2, n_patients * 2),
            'PLATELETS': np.random.normal(250, 50, n_patients * 2),
            'CREATININE': np.random.normal(1.0, 0.3, n_patients * 2),
            'ALT': np.random.normal(25, 10, n_patients * 2),
            'AST': np.random.normal(23, 9, n_patients * 2)
        })
        
        datasets = {
            'demographics': demographics,
            'baseline_measurements': baseline_measurements,
            'followup_measurements': followup_measurements,
            'all_measurements': all_measurements,
            'adverse_events': adverse_events_df,
            'lab_results': lab_results
        }
        
        # Save sample data to CSV files for demonstration
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        for name, df in datasets.items():
            csv_path = os.path.join(data_dir, f"{name}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Sample data saved: {csv_path}")
        
        return datasets
    
    def run_complete_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run complete analysis workflow demonstrating SAS modernization.
        
        Args:
            datasets: Dictionary of datasets to analyze
            
        Returns:
            Dictionary of analysis results
        """
        logger.info("Starting complete analysis workflow")
        
        results = {}
        
        # 1. Data Processing Phase (replaces SAS DATA steps)
        logger.info("Phase 1: Data Processing")
        
        # Clean demographics data
        self.data_processor.datasets = datasets.copy()  # Load datasets
        
        clean_demographics = self.data_processor.clean_data('demographics')
        
        # Create derived variables (replaces SAS derivations)
        age_derivations = {
            'AGE_GROUP': 'pd.cut(df["AGE"], bins=[0, 50, 65, 80, 100], labels=["<50", "50-64", "65-79", "80+"])',
            'ELDERLY': 'df["AGE"] >= 65'
        }
        self.data_processor.derive_variables('demographics', age_derivations)
        
        # Merge datasets (replaces SAS MERGE)
        merged_baseline = self.data_processor.merge_datasets(
            'demographics', 'baseline_measurements', 
            'PATIENT_ID', result_name='analysis_baseline'
        )
        
        # 2. Statistical Analysis Phase (replaces SAS PROC steps)
        logger.info("Phase 2: Statistical Analysis")
        
        # Descriptive statistics (replaces PROC MEANS)
        numeric_vars = ['AGE', 'HEIGHT_CM', 'WEIGHT_KG', 'BMI', 'SYSTOLIC_BP', 'DIASTOLIC_BP']
        means_results = self.statistical_analyzer.proc_means(
            merged_baseline, variables=numeric_vars, by_variables=['TREATMENT_GROUP']
        )
        results['descriptive_stats'] = means_results
        
        # Frequency analysis (replaces PROC FREQ)
        categorical_vars = ['SEX', 'RACE', 'AGE_GROUP']
        freq_results = self.statistical_analyzer.proc_freq(
            merged_baseline, variables=categorical_vars, by_variables=['TREATMENT_GROUP']
        )
        results['frequency_analysis'] = freq_results
        
        # Statistical tests (replaces PROC TTEST)
        for var in ['AGE', 'SYSTOLIC_BP', 'DIASTOLIC_BP']:
            ttest_result = self.statistical_analyzer.proc_ttest(
                merged_baseline, var, 'TREATMENT_GROUP'
            )
            results[f'ttest_{var}'] = ttest_result
        
        # Correlation analysis (replaces PROC CORR)
        corr_vars = ['AGE', 'BMI', 'SYSTOLIC_BP', 'DIASTOLIC_BP', 'CHOLESTEROL']
        corr_results = self.statistical_analyzer.proc_corr(merged_baseline, corr_vars)
        results['correlation_matrix'] = corr_results
        
        # Create clinical summary table
        summary_table = self.statistical_analyzer.create_summary_table(
            merged_baseline, variables=numeric_vars + categorical_vars,
            group_variable='TREATMENT_GROUP', include_tests=True
        )
        results['clinical_summary'] = summary_table
        
        logger.info("Analysis completed successfully")
        return results
    
    def generate_comprehensive_reports(self, datasets: Dict[str, pd.DataFrame], 
                                     results: Dict):
        """
        Generate comprehensive reports (replaces SAS ODS output).
        
        Args:
            datasets: Dictionary of datasets
            results: Analysis results
        """
        logger.info("Generating comprehensive reports")
        
        # 1. Generate Excel workbook (replaces ODS EXCEL)
        excel_data = {
            'Demographics': datasets['demographics'],
            'Baseline_Measurements': datasets['baseline_measurements'],
            'Adverse_Events': datasets['adverse_events'],
            'Descriptive_Stats': results['descriptive_stats'],
            'Clinical_Summary': results['clinical_summary']
        }
        
        excel_path = os.path.join(self.output_dir, 'clinical_trial_analysis.xlsx')
        self.report_generator.export_to_excel(excel_data, excel_path)
        
        # 2. Generate HTML dashboard (replaces ODS HTML)
        dashboard_path = os.path.join(self.output_dir, 'analysis_dashboard.html')
        self.report_generator.create_summary_dashboard(
            datasets, self.statistical_analyzer.get_all_results(), dashboard_path
        )
        
        # 3. Generate detailed analysis report
        analysis_data = {
            'summary_stats': [
                {'value': len(datasets['demographics']), 'label': 'Total Patients'},
                {'value': len(datasets['adverse_events']), 'label': 'Adverse Events'},
                {'value': f"{len(datasets['demographics'][datasets['demographics']['TREATMENT_GROUP'] == 'TREATMENT'])}", 'label': 'Treatment Group'},
                {'value': f"{len(datasets['demographics'][datasets['demographics']['TREATMENT_GROUP'] == 'PLACEBO'])}", 'label': 'Placebo Group'}
            ],
            'sections': [
                {
                    'title': 'Patient Demographics',
                    'description': 'Baseline demographic characteristics of study population',
                    'dataframe': results['clinical_summary']
                },
                {
                    'title': 'Statistical Test Results',
                    'description': 'Summary of statistical comparisons between treatment groups',
                    'content': self._format_test_results(results)
                }
            ]
        }
        
        detailed_report_path = os.path.join(self.output_dir, 'detailed_analysis_report.html')
        self.report_generator.create_html_report(
            title="Clinical Trial Analysis Report",
            data=analysis_data,
            output_path=detailed_report_path,
            subtitle="Comprehensive Statistical Analysis Results"
        )
        
        logger.info("All reports generated successfully")
        
        # Print report summary
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETE")
        print("="*60)
        print(f"Excel Report: {excel_path}")
        print(f"HTML Dashboard: {dashboard_path}")
        print(f"Detailed Report: {detailed_report_path}")
        print("="*60)
    
    def _format_test_results(self, results: Dict) -> str:
        """Format statistical test results for HTML display."""
        content = "<h3>Statistical Test Summary</h3>"
        
        # Format t-test results
        for key, result in results.items():
            if key.startswith('ttest_'):
                var_name = key.replace('ttest_', '')
                content += f"""
                <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #ddd;'>
                    <h4>{var_name.replace('_', ' ').title()}</h4>
                    <p><strong>Test Type:</strong> {result['test_type']}</p>
                    <p><strong>T-statistic:</strong> {result['t_statistic']:.4f}</p>
                    <p><strong>P-value:</strong> {result['p_value']:.4f}</p>
                    <p><strong>Result:</strong> {'Statistically Significant' if result['significant'] else 'Not Significant'} (α = {result['alpha']})</p>
                    <p><strong>Group 1 Mean:</strong> {result['mean1']:.2f} (n={result['n1']})</p>
                    <p><strong>Group 2 Mean:</strong> {result['mean2']:.2f} (n={result['n2']})</p>
                </div>
                """
        
        return content


def main():
    """Main function demonstrating the SAS modernization prototype."""
    print("SAS Process Modernization Prototype")
    print("=" * 40)
    print("Demonstrating Python-based modernization of legacy SAS clinical trial processes")
    
    # Initialize processor
    processor = ClinicalTrialProcessor()
    
    # Generate or load sample data
    print("\n1. Generating sample clinical trial data...")
    datasets = processor.generate_sample_data(n_patients=500)
    
    # Run complete analysis
    print("\n2. Running comprehensive statistical analysis...")
    results = processor.run_complete_analysis(datasets)
    
    # Generate reports
    print("\n3. Generating comprehensive reports...")
    processor.generate_comprehensive_reports(datasets, results)
    
    print("\nPrototype demonstration completed successfully!")
    print("\nThis prototype demonstrates:")
    print("- Modern Python-based data processing (replacing SAS DATA steps)")
    print("- Comprehensive statistical analysis (replacing SAS PROC steps)")
    print("- Professional report generation (replacing SAS ODS output)")
    print("- Rigorous testing framework for accuracy and reliability")
    print("- Improved performance, maintainability, and collaboration capabilities")


if __name__ == "__main__":
    main()