import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime

class ProfitMaximizingPipeline:
    def __init__(self, call_cost=8, contract_profit=80):
        self.call_cost = call_cost
        self.contract_profit = contract_profit
        self.preprocessor = None
        self.model = None
        self.threshold = None
        self.profit_data = None
        self.feature_names = None
        self.feature_categories = None
    
    def load_data(self, filepath):
        print("Loading data from:", filepath)
        df = pd.read_csv(filepath, delimiter=';')
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip('"')
        
        df['target'] = (df['y'] == 'yes').astype(int)
        df = df.drop(['duration', 'y'], axis=1)
        
        return df
    
    def engineer_features(self, df):
        print("Engineering features...")
        
        df_engineered = df.copy()
        
        df_engineered['poutcome_success'] = (df_engineered['poutcome'] == 'success').astype(int)
        df_engineered['poutcome_failure'] = (df_engineered['poutcome'] == 'failure').astype(int)
        df_engineered['contact_cellular'] = (df_engineered['contact'] == 'cellular').astype(int)
        
        high_conv_months = ['mar', 'dec', 'sep', 'oct']
        df_engineered['month_high_conv'] = df_engineered['month'].isin(high_conv_months).astype(int)
        
        high_conv_jobs = ['student', 'retired']
        low_conv_jobs = ['blue-collar']
        df_engineered['job_high_conv'] = df_engineered['job'].isin(high_conv_jobs).astype(int)
        df_engineered['job_low_conv'] = df_engineered['job'].isin(low_conv_jobs).astype(int)
        
        df_engineered['education_university'] = (df_engineered['education'] == 'university.degree').astype(int)
        try:
            df_engineered['education_illiterate'] = (df_engineered['education'] == 'illiterate').astype(int)
        except:
            df_engineered['education_illiterate'] = 0
        
        df_engineered['cellular_success'] = df_engineered['contact_cellular'] * df_engineered['poutcome_success']
        df_engineered['cellular_high_month'] = df_engineered['contact_cellular'] * df_engineered['month_high_conv']
        df_engineered['success_high_month'] = df_engineered['poutcome_success'] * df_engineered['month_high_conv']
        df_engineered['top_combo'] = df_engineered['contact_cellular'] * df_engineered['poutcome_success'] * df_engineered['month_high_conv']
        
        df_engineered['age_young'] = (df_engineered['age'] <= 25).astype(int)
        df_engineered['age_senior'] = (df_engineered['age'] >= 60).astype(int)
        
        df_engineered['euribor3m_low'] = (df_engineered['euribor3m'] < 2.0).astype(int)
        df_engineered['emp_var_rate_neg'] = (df_engineered['emp.var.rate'] < 0).astype(int)
        
        df_engineered['previously_contacted'] = (df_engineered['pdays'] != 999).astype(int)
        
        return df_engineered
    
    def create_preprocessing_pipeline(self):
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                              'contact', 'month', 'day_of_week', 'poutcome']
        
        numeric_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                          'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        
        binary_features = [
            'poutcome_success', 'poutcome_failure',
            'contact_cellular',
            'month_high_conv',
            'job_high_conv', 'job_low_conv',
            'education_university', 'education_illiterate',
            'cellular_success', 'cellular_high_month', 'success_high_month', 'top_combo',
            'age_young', 'age_senior',
            'euribor3m_low', 'emp_var_rate_neg',
            'previously_contacted'
        ]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features),

                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                
                ('bin', 'passthrough', binary_features)
            ],
            verbose_feature_names_out=False
        )
        
        self.feature_categories = {
            'categorical': categorical_features,
            'numeric': numeric_features,
            'binary': binary_features
        }
        
        return preprocessor
    
    def find_optimal_threshold(self, y_true, y_proba):
        print("Finding optimal threshold...")
        
        thresholds = np.arange(0.01, 0.99, 0.01)
        profits = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            calls_made = y_pred.sum()
            if calls_made == 0:
                continue
                
            successful_conversions = np.logical_and(y_pred, y_true).sum()
            
            call_cost = calls_made * self.call_cost
            conversion_profit = successful_conversions * self.contract_profit
            net_profit = conversion_profit - call_cost
            
            total_customers = len(y_true)
            total_potential_conversions = y_true.sum()
            
            call_reduction = 1 - (calls_made / total_customers)
            conversion_rate = successful_conversions / calls_made if calls_made > 0 else 0
            conversion_capture = successful_conversions / total_potential_conversions if total_potential_conversions > 0 else 0
            
            profits.append({
                'threshold': threshold,
                'calls_made': calls_made,
                'successful_conversions': successful_conversions,
                'conversion_rate': conversion_rate,
                'call_cost': call_cost,
                'conversion_profit': conversion_profit,
                'net_profit': net_profit,
                'call_reduction': call_reduction,
                'conversion_capture': conversion_capture
            })
        
        profit_data = pd.DataFrame(profits)
        
        optimal_idx = profit_data['net_profit'].idxmax()
        optimal_threshold = profit_data.loc[optimal_idx, 'threshold']
        
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Maximum profit: €{profit_data.loc[optimal_idx, 'net_profit']:.2f}")
        print(f"Call reduction: {profit_data.loc[optimal_idx, 'call_reduction']*100:.1f}%")
        print(f"Conversion rate: {profit_data.loc[optimal_idx, 'conversion_rate']*100:.1f}%")
        print(f"Conversion capture: {profit_data.loc[optimal_idx, 'conversion_capture']*100:.1f}%")
        
        return optimal_threshold, profit_data
    
    def fit(self, data_path):
        df = self.load_data(data_path)
        df = self.engineer_features(df)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.preprocessor = self.create_preprocessing_pipeline()
        
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42
            ))
        ])
        
        print("Training XGBoost model...")
        self.model.fit(X_train, y_train)
        
        print("\nEvaluating model...")
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}")
        
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        self.threshold, self.profit_data = self.find_optimal_threshold(y_test, y_prob)
        
        self.analyze_feature_importance()
        
        return self
    
    def analyze_feature_importance(self):
        print("\nAnalyzing feature importance...")
        
        try:
            categorical_cols = self.feature_categories['categorical']
            numeric_cols = self.feature_categories['numeric']
            binary_cols = self.feature_categories['binary']
            
            ohe = self.model.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot']
            categorical_features = ohe.get_feature_names_out(categorical_cols)
            
            self.feature_names = list(categorical_features) + numeric_cols + binary_cols
            
            importances = self.model.named_steps['classifier'].feature_importances_
            
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("Top 20 most important features:")
            print(feature_importance.head(20))
            
            return feature_importance
        
        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            return None
    
    def predict(self, X):
        if self.model is None or self.threshold is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        probas = self.model.predict_proba(X)[:, 1]
        
        predictions = (probas >= self.threshold).astype(int)
        
        results = pd.DataFrame({
            'call_probability': probas,
            'call_decision': predictions
        })
        
        total = len(results)
        to_call = results['call_decision'].sum()
        call_reduction = 1 - (to_call / total)
        
        print(f"Total customers: {total}")
        print(f"Customers to call: {to_call} ({to_call/total:.1%})")
        print(f"Call reduction: {call_reduction:.1%}")
        
        return results
    
    def score_customers(self, customer_data, output_csv=None):
        if isinstance(customer_data, str):
            input_file = customer_data
            print(f"Loading customer data from: {input_file}")
            customer_data = pd.read_csv(customer_data, delimiter=';')
        else:
            input_file = "customer_data"
        
        original_data = customer_data.copy()
        
        has_y_column = 'y' in original_data.columns
        if has_y_column:
            y_values = original_data['y'].copy()
        
        proc_data = original_data.copy()
        
        if 'duration' in proc_data.columns:
            proc_data = proc_data.drop('duration', axis=1)
        
        if 'y' in proc_data.columns:
            proc_data = proc_data.drop('y', axis=1)
            
        processed_data = self.engineer_features(proc_data)
        
        results = self.predict(processed_data)
        
        scored_customers = original_data.copy()
        scored_customers['conversion_probability'] = results['call_probability']
        scored_customers['should_call'] = results['call_decision']
        
        customers_to_call = scored_customers[scored_customers['should_call'] == 1].copy()
        
        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(os.path.basename(input_file))[0] if isinstance(input_file, str) else "customers"
            output_csv = f"{base_filename}_to_call_{timestamp}.csv"
        
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        customers_to_call.to_csv(output_csv, index=False, sep=';')
        print(f"Filtered dataset with {len(customers_to_call)} customers to call saved to: {output_csv}")
        
        full_output = os.path.splitext(output_csv)[0] + "_all_scored.csv"
        scored_customers.to_csv(full_output, index=False, sep=';')
        print(f"Complete scored dataset saved to: {full_output}")
        
        return scored_customers, customers_to_call
    
    def export_filtered_dataset(self, data_path, output_path=None):
        print(f"Processing dataset: {data_path}")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(os.path.basename(data_path))[0]
            output_path = f"{base_filename}_filtered_{timestamp}.csv"
        
        _, customers_to_call = self.score_customers(data_path, output_path)
        
        full_output = os.path.splitext(output_path)[0] + "_all_scored.csv"
        
        return output_path, full_output

if __name__ == "__main__":
    pipeline = ProfitMaximizingPipeline(call_cost=8, contract_profit=80)
    
    pipeline.fit('bank-additional-full.csv')
    
    output_csv, full_csv = pipeline.export_filtered_dataset('bank-additional-full.csv')
    
    print("\nPipeline execution completed!")
    print(f"Filtered dataset: {output_csv}")
    print(f"Full dataset: {full_csv}")
    
    call_count = len(pd.read_csv(output_csv, delimiter=';'))
    total_count = len(pd.read_csv(full_csv, delimiter=';'))
    reduction = 1 - (call_count / total_count)
    
    print(f"\nResults Summary:")
    print(f"- Original customer count: {total_count}")
    print(f"- Customers to call: {call_count}")
    print(f"- Call reduction: {reduction:.1%}")
    print(f"- Estimated cost savings: €{(total_count - call_count) * 8:.2f}")