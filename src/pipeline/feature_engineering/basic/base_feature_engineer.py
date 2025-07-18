import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BaseFeatureEngineer:
    """מחלקת בסיס משופרת להנדסת תכונות עם קידוד מלא"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_stats = {}
    
    def identify_all_categorical_columns(self, df):
        """זיהוי כל העמודות הקטגוריות/טקסטואליות ללא יצירת הגבלות"""
        categorical_columns = []
        
        # הגדרת עמודות שאסור לקודד (רק עמודות שבאמת לא צריכות קידוד)
        protected_columns = [
            'is_malicious',  # target column
            'date',          # date columns שצריכות טיפול מיוחד
            'timestamp'      # timestamp columns
        ]
        
        # בדיקת כל עמודה
        for col in df.columns:
            if col in protected_columns:
                continue
                
            # בדיקת סוג הנתונים
            dtype = df[col].dtype
            
            # עמודות טקסט מפורשות
            if dtype in ['object', 'category']:
                categorical_columns.append(col)
                continue
            
            # עמודות מספריות שעשויות להיות קטגוריות
            if dtype in ['int64', 'int32', 'float64', 'float32']:
                unique_values = df[col].nunique()
                total_values = len(df[col])
                
                # אם יש מעט ערכים ייחודיים ביחס לכמות הנתונים
                if unique_values <= 50 and unique_values < total_values * 0.1:
                    sample_values = df[col].dropna().unique()[:10]
                    
                    # אם הערכים נראים כמו קטגוריות
                    if any(isinstance(val, (str, bool)) for val in sample_values):
                        categorical_columns.append(col)
                    elif unique_values <= 10:  # מספרים עם מעט ערכים
                        categorical_columns.append(col)
        
        return categorical_columns
    
    def _handle_categorical_missing_values(self, series):
        """טיפול בערכים חסרים בעמודות קטגוריות"""
        if pd.api.types.is_categorical_dtype(series):
            # אם זה Categorical, נוסיף את הקטגוריה 'Missing' לרשימת הקטגוריות
            if 'Missing' not in series.cat.categories:
                series = series.cat.add_categories(['Missing'])
            return series.fillna('Missing')
        else:
            # אם זה לא Categorical, נטפל בדרך הרגילה
            return series.fillna('Missing')
    
    def encode_categorical_variables(self, df, target_col='is_malicious'):
        """קידוד מלא לכל המשתנים הקטגוריים"""
        print("Starting comprehensive categorical encoding...")
        df_processed = df.copy()
        
        # זיהוי כל העמודות הקטגוריות
        categorical_columns = self.identify_all_categorical_columns(df)
        
        # הוספת כל עמודות object שלא נתפסו
        object_columns = df_processed.select_dtypes(include=['object']).columns
        for col in object_columns:
            if col not in categorical_columns and col not in ['is_malicious', 'date', 'timestamp']:
                categorical_columns.append(col)
        
        print(f"Found {len(categorical_columns)} categorical columns to encode:")
        print(categorical_columns[:10])  # הצגת 10 הראשונות
        
        encoded_features = []
        columns_to_drop = []  # רשימת עמודות למחיקה
        
        for col in categorical_columns:
            if col not in df_processed.columns:
                continue
                
            print(f"Processing column: {col}")
            
            # טיפול בערכים חסרים - תיקון הבעיה העיקרית
            df_processed[col] = self._handle_categorical_missing_values(df_processed[col])
            
            # המרה לטקסט (אם זה לא Categorical)
            if not pd.api.types.is_categorical_dtype(df_processed[col]):
                df_processed[col] = df_processed[col].astype(str)
            else:
                # אם זה Categorical, נמיר לטקסט
                df_processed[col] = df_processed[col].astype(str)
            
            unique_values = df_processed[col].nunique()
            
            try:
                # שיטת קידוד בהתאם לכמות הערכים הייחודיים
                if unique_values == 1:
                    # עמודה עם ערך יחיד - נסמן אותה לזריקה
                    print(f"  Column {col} has only one unique value, skipping")
                    columns_to_drop.append(col)
                    continue
                    
                elif unique_values == 2:
                    # קידוד בינארי פשוט
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_processed[f'{col}_binary'] = self.encoders[col].fit_transform(df_processed[col])
                    encoded_features.append(f'{col}_binary')
                    columns_to_drop.append(col)
                    
                elif unique_values <= 10:
                    # One-hot encoding לעמודות עם cardinality נמוך
                    dummies = pd.get_dummies(df_processed[col], prefix=f'{col}_cat')
                    df_processed = pd.concat([df_processed, dummies], axis=1)
                    encoded_features.extend(dummies.columns.tolist())
                    
                    # גם label encoding
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_processed[f'{col}_label'] = self.encoders[col].fit_transform(df_processed[col])
                    encoded_features.append(f'{col}_label')
                    columns_to_drop.append(col)
                    
                elif unique_values <= 50:
                    # Label encoding + Target encoding (אם יש target)
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_processed[f'{col}_label'] = self.encoders[col].fit_transform(df_processed[col])
                    encoded_features.append(f'{col}_label')
                    
                    # Target encoding
                    if target_col in df_processed.columns:
                        target_mean = df_processed.groupby(col)[target_col].mean()
                        df_processed[f'{col}_target'] = df_processed[col].map(target_mean)
                        encoded_features.append(f'{col}_target')
                    
                    # Frequency encoding
                    freq_map = df_processed[col].value_counts().to_dict()
                    df_processed[f'{col}_freq'] = df_processed[col].map(freq_map)
                    encoded_features.append(f'{col}_freq')
                    columns_to_drop.append(col)
                    
                else:
                    # עמודות עם cardinality גבוה - רק frequency ו-target encoding
                    freq_map = df_processed[col].value_counts().to_dict()
                    df_processed[f'{col}_freq'] = df_processed[col].map(freq_map)
                    encoded_features.append(f'{col}_freq')
                    
                    if target_col in df_processed.columns:
                        target_mean = df_processed.groupby(col)[target_col].mean()
                        df_processed[f'{col}_target'] = df_processed[col].map(target_mean)
                        encoded_features.append(f'{col}_target')
                    
                    # Hash encoding לעמודות עם cardinality גבוה מאוד
                    if unique_values > 100:
                        df_processed[f'{col}_hash'] = df_processed[col].apply(lambda x: hash(str(x)) % 1000)
                        encoded_features.append(f'{col}_hash')
                    
                    columns_to_drop.append(col)
                
                print(f"  Successfully encoded {col} with {unique_values} unique values")
                
            except Exception as e:
                print(f"  Error encoding column {col}: {str(e)}")
                # fallback - לפחות label encoding
                try:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df_processed[f'{col}_fallback'] = self.encoders[col].fit_transform(df_processed[col])
                    encoded_features.append(f'{col}_fallback')
                    columns_to_drop.append(col)
                except:
                    print(f"  Failed to encode {col} even with fallback method")
        
        # מחיקת כל העמודות המקוריות שקודדו
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if existing_columns_to_drop:
            df_processed = df_processed.drop(columns=existing_columns_to_drop)
            print(f"Dropped {len(existing_columns_to_drop)} original categorical columns")
        
        print(f"\nCategorical encoding completed! Created {len(encoded_features)} new encoded features")
        print(f"Removed {len(existing_columns_to_drop)} original categorical columns")
        return df_processed

    def handle_special_text_columns(self, df):
        """טיפול מיוחד בעמודות טקסט מורכבות"""
        df_processed = df.copy()
        
        # זיהוי עמודות עם טקסט מורכב - בדיקה מחודשת כי יכול להיות שנותרו עמודות
        text_columns = []
        for col in df_processed.columns:
            if col not in ['is_malicious', 'date', 'timestamp']:
                # בדיקה אם זה עמודת טקסט שנותרה
                if df_processed[col].dtype == 'object':
                    text_columns.append(col)
        
        columns_to_drop = []  # רשימת עמודות למחיקה
        
        print(f"Found {len(text_columns)} remaining text columns to process")
        
        for col in text_columns:
            if col in df_processed.columns:
                # בדיקת אורך הטקסט
                try:
                    text_lengths = df_processed[col].astype(str).str.len()
                    df_processed[f'{col}_length'] = text_lengths
                    
                    # כמות מילים
                    df_processed[f'{col}_word_count'] = df_processed[col].astype(str).str.split().str.len()
                    
                    # האם מכיל מספרים
                    df_processed[f'{col}_has_numbers'] = df_processed[col].astype(str).str.contains(r'\d', na=False).astype(int)
                    
                    # האם מכיל תווים מיוחדים
                    df_processed[f'{col}_has_special'] = df_processed[col].astype(str).str.contains(r'[^a-zA-Z0-9\s]', na=False).astype(int)
                    
                    # תדירות תווים
                    df_processed[f'{col}_char_diversity'] = df_processed[col].astype(str).apply(lambda x: len(set(x)) if x else 0)
                    
                    # אם יצרנו תכונות חדשות, נוסיף את העמודה המקורית לרשימת המחיקה
                    columns_to_drop.append(col)
                    print(f"Processed and will drop text column: {col}")
                    
                except Exception as e:
                    print(f"Error processing text column {col}: {str(e)}")
        
        # מחיקת העמודות המקוריות שעובדו
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if existing_columns_to_drop:
            df_processed = df_processed.drop(columns=existing_columns_to_drop)
            print(f"Dropped {len(existing_columns_to_drop)} original text columns after feature extraction")
        
        # בדיקה סופית ומחיקת כל עמודת object שנותרה (פרט לעמודות מוגנות)
        remaining_objects = df_processed.select_dtypes(include=['object']).columns
        final_drops = [col for col in remaining_objects if col not in ['is_malicious', 'date', 'timestamp']]
        
        if final_drops:
            print(f"Final cleanup: dropping remaining object columns: {final_drops}")
            df_processed = df_processed.drop(columns=final_drops)
        
        return df_processed
        
        # מחיקת העמודות המקוריות שעובדו
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if existing_columns_to_drop:
            df_processed = df_processed.drop(columns=existing_columns_to_drop)
            print(f"Dropped {len(existing_columns_to_drop)} original text columns after feature extraction")
        
        return df_processed
        
    def apply_statistical_transforms(self, df):
            """החלת טרנספורמציות סטטיסטיות משופרות"""
            df_processed = df.copy()
            
            # זיהוי עמודות מספריות לטרנספורמציה
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            
            # הסרת עמודות מיוחדות
            exclude_cols = ['is_malicious', 'date', 'timestamp']
            transform_columns = [col for col in numeric_columns if col not in exclude_cols]
            
            print(f"Applying statistical transforms to {len(transform_columns)} numeric columns")
            
            for col in transform_columns:
                if col in df_processed.columns:
                    # בדיקת תקינות הנתונים
                    if df_processed[col].isna().all() or df_processed[col].nunique() <= 1:
                        print(f"Skipping column {col} - insufficient data")
                        continue
                        
                    try:
                        # טרנספורמציה לוגריתמית
                        if (df_processed[col] >= 0).all():
                            df_processed[f'{col}_log'] = np.log1p(df_processed[col])
                        
                        # טרנספורמציה שורש
                        if (df_processed[col] >= 0).all():
                            df_processed[f'{col}_sqrt'] = np.sqrt(df_processed[col])
                        
                        # Z-score (רק אם יש שונות)
                        if df_processed[col].std() > 0:
                            df_processed[f'{col}_zscore'] = stats.zscore(df_processed[col])
                        
                        # בינארי (מעל/מתחת לממוצע)
                        df_processed[f'{col}_above_mean'] = (df_processed[col] > df_processed[col].mean()).astype(int)
                        
                        # רבעונים עם טיפול טוב יותר
                        unique_values = df_processed[col].nunique()
                        
                        if unique_values >= 4:
                            try:
                                df_processed[f'{col}_quartile'] = pd.qcut(
                                    df_processed[col], 
                                    q=4, 
                                    duplicates='drop'
                                )
                            except:
                                # fallback
                                median_val = df_processed[col].median()
                                df_processed[f'{col}_quartile'] = np.where(
                                    df_processed[col] <= median_val, 0, 1
                                )
                        else:
                            df_processed[f'{col}_quartile'] = 0
                            
                    except Exception as e:
                        print(f"Error transforming column {col}: {str(e)}")
            
            print("Statistical transformations completed")
            return df_processed
        
    def get_comprehensive_feature_summary(self, df):
            """סיכום מקיף של כל התכונות"""
            
            feature_groups = {
                'original_features': [],
                'time_features': [],
                'printing_features': [],
                'burning_features': [],
                'employee_features': [],
                'access_features': [],
                'interaction_features': [],
                'encoded_categorical': [],
                'text_analysis': [],
                'statistical_transforms': [],
                'binary_features': [],
                'other_features': []
            }
            
            # סיווג התכונות
            for col in df.columns:
                if any(x in col for x in ['year', 'month', 'day', 'hour', 'season', 'quarter']):
                    feature_groups['time_features'].append(col)
                elif 'print' in col:
                    feature_groups['printing_features'].append(col)
                elif 'burn' in col:
                    feature_groups['burning_features'].append(col)
                elif any(x in col for x in ['employee', 'seniority', 'department']):
                    feature_groups['employee_features'].append(col)
                elif any(x in col for x in ['entry', 'exit', 'presence', 'access']):
                    feature_groups['access_features'].append(col)
                elif any(x in col for x in ['ratio', 'per_', 'interaction']):
                    feature_groups['interaction_features'].append(col)
                elif any(x in col for x in ['_label', '_target', '_freq', '_hash', '_cat_', '_binary', '_fallback']):
                    feature_groups['encoded_categorical'].append(col)
                elif any(x in col for x in ['_length', '_word_count', '_has_numbers', '_has_special', '_char_diversity']):
                    feature_groups['text_analysis'].append(col)
                elif any(x in col for x in ['_log', '_sqrt', '_zscore', '_quartile', '_above_mean']):
                    feature_groups['statistical_transforms'].append(col)
                elif df[col].dtype == 'bool' or df[col].nunique() == 2:
                    feature_groups['binary_features'].append(col)
                else:
                    feature_groups['other_features'].append(col)
            
            # יצירת סיכום
            summary = {}
            total_features = 0
            
            for group, features in feature_groups.items():
                summary[group] = {
                    'count': len(features),
                    'features': features[:10],  # הצגת 10 הראשונות
                    'sample_features': features[:5] if features else []
                }
                total_features += len(features)
            
            summary['total_features'] = total_features
            summary['categorical_columns_processed'] = len(feature_groups['encoded_categorical'])
            summary['text_analysis_features'] = len(feature_groups['text_analysis'])
            summary['statistical_features'] = len(feature_groups['statistical_transforms'])
            
            return summary