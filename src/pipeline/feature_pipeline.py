# feature_pipeline.py - Unified Feature Pipeline Module
import pandas as pd
from .feature_engineering.feature_engineer_manager.categorical_encoder import CategoricalEncoder
from .feature_engineering.feature_engineer_manager.statistical_transformer import StatisticalTransformer

class FeatureEngineer:
    """מנהל pipeline של הנדסת תכונות - כולל גם עיבוד וקידוד נתונים"""
    
    def __init__(self):
        self.categorical_encoder = CategoricalEncoder()
        self.statistical_transformer = StatisticalTransformer()
        self.basic_types = ['time', 'printing', 'burning', 'employee', 'access', 'interaction']
        self.advanced_types = ['behavioral', 'temporal', 'risk_profile', 'anomaly', 'advanced_interaction']

    # ==================== Feature Engineering Methods ====================
    
    def create_all_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """יצירת כל התכונות הבסיסיות"""
        print("Starting basic feature engineering...")
        for feature_type in self.basic_types:
            try:
                df = self.factory.create_features_by_type(df, feature_type)
                print(f"{feature_type} features created")
            except Exception as e:
                print(f"Error in {feature_type}: {e}")
        return df
    
    def create_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """יצירת כל התכונות המתקדמות"""
        print("Starting advanced feature engineering...")
        for feature_type in self.advanced_types:
            try:
                df = self.factory.create_features_by_type(df, feature_type)
            except Exception as e:
                print(f"Error in {feature_type}: {e}")
        
        # תכונות מתקדמות מיוחדות
        specialized_methods = [
            ('ratio', 'create_ratio_features'),
            ('polynomial', 'create_polynomial_features')
        ]
        
        for name, method in specialized_methods:
            try:
                df = self.factory.safe_engineer_call('advanced_interaction', method, df)
            except Exception as e:
                print(f"Error in {name}: {e}")
        
        print("Advanced feature engineering completed!")
        return df
    
    def remove_original_columns(self, df: pd.DataFrame, columns_to_remove=None) -> pd.DataFrame:
        """הסרת עמודות מקוריות לפני הקידוד"""
        if columns_to_remove is None:
            columns_to_remove = ['employee_origin_country', 'country_name', 'first_entry_time', 'last_exit_time', 'modification_details', 'row_modified']
        
        df_processed = df.copy()
        existing_columns = [col for col in columns_to_remove if col in df_processed.columns]
        
        if existing_columns:
            df_processed = df_processed.drop(columns=existing_columns)
            print(f"Removed original columns before encoding: {existing_columns}")
        else:
            print("No original columns found to remove")
        
        return df_processed
    
    def standardize_data_types(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """סטנדרטיזציה של טיפוסי נתונים"""
        # המרת עמודות boolean לנומריות
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(int)
        
        # המרת עמודות object שהן למעשה נומריות
        for col in df.select_dtypes(include=['object']).columns:
            if col != target_col:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        return df