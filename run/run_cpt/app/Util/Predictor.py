import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

class PatternPredictor:
    def __init__(self):
        self.classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=2,
            callbacks=[self._training_callback()]
        )
        
        self.regressor = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=2,
            callbacks=[self._training_callback()]
        )
        
        self.scaler = StandardScaler()
        self.training_progress = {'classifier': [], 'regressor': []}
        
    def _training_callback(self):
        def callback(env):
            model_type = 'classifier' if isinstance(env.model, XGBClassifier) else 'regressor'
            iteration = len(env.evaluation_result_list)
            eval_result = env.evaluation_result_list[0][1]
            
            self.training_progress[model_type].append({
                'iteration': iteration,
                'metric': eval_result
            })
            
            if iteration % 10 == 0:  # Print every 10 iterations
                print(f"{model_type.capitalize()} - Iteration {iteration}: {env.evaluation_result_list[0][0]} = {eval_result:.4f}")
                
        return callback

    def fit(self, X, y_binary, y_returns, test_size=0.2):
        """Train models with progress tracking"""
        # Reset training progress
        self.training_progress = {'classifier': [], 'regressor': []}
        
        # Split data
        X_train, X_test, y_bin_train, y_bin_test, y_ret_train, y_ret_test = train_test_split(
            X, y_binary, y_returns, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\nTraining Classifier...")
        self.classifier.fit(
            X_train_scaled, y_bin_train,
            eval_set=[(X_test_scaled, y_bin_test)],
            eval_metric=['logloss', 'error'],
            verbose=False
        )
        
        print("\nTraining Regressor...")
        self.regressor.fit(
            X_train_scaled, y_ret_train,
            eval_set=[(X_test_scaled, y_ret_test)],
            eval_metric='rmse',
            verbose=False
        )
        
        # Calculate metrics
        y_bin_pred = self.classifier.predict(X_test_scaled)
        y_ret_pred = self.regressor.predict(X_test_scaled)
        
        metrics = {
            'classification': {
                'precision': precision_score(y_bin_test, y_bin_pred),
                'recall': recall_score(y_bin_test, y_bin_pred),
                'f1': f1_score(y_bin_test, y_bin_pred),
                'confusion_matrix': confusion_matrix(y_bin_test, y_bin_pred),
                'win_rate': np.mean(y_bin_pred),
                'actual_win_rate': np.mean(y_bin_test)
            },
            'regression': {
                'mse': np.mean((y_ret_test - y_ret_pred) ** 2),
                'mae': np.mean(np.abs(y_ret_test - y_ret_pred)),
                'avg_predicted_return': np.mean(y_ret_pred),
                'avg_actual_return': np.mean(y_ret_test)
            }
        }
        
        return metrics

    def plot_metrics(self, metrics):
        """Create interactive Plotly visualizations"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'Classification Metrics',
                          'Training Progress', 'Return Metrics')
        )
        
        # Confusion Matrix
        conf_matrix = metrics['classification']['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=conf_matrix,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                text=conf_matrix,
                texttemplate="%{z}",
                textfont={"size": 20},
                colorscale='Blues'
            ),
            row=1, col=1
        )
        
        # Classification Metrics
        metrics_to_plot = {
            'Precision': metrics['classification']['precision'],
            'Recall': metrics['classification']['recall'],
            'F1 Score': metrics['classification']['f1'],
            'Win Rate': metrics['classification']['win_rate']
        }
        fig.add_trace(
            go.Bar(
                x=list(metrics_to_plot.keys()),
                y=list(metrics_to_plot.values()),
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Training Progress
        fig.add_trace(
            go.Scatter(
                x=[x['iteration'] for x in self.training_progress['classifier']],
                y=[x['metric'] for x in self.training_progress['classifier']],
                name='Classifier Loss',
                mode='lines'
            ),
            row=2, col=1
        )
        
        # Return Metrics
        return_metrics = {
            'MSE': metrics['regression']['mse'],
            'MAE': metrics['regression']['mae'],
            'Avg Pred Return': metrics['regression']['avg_predicted_return'],
            'Avg Act Return': metrics['regression']['avg_actual_return']
        }
        fig.add_trace(
            go.Bar(
                x=list(return_metrics.keys()),
                y=list(return_metrics.values()),
                marker_color='lightgreen'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, width=1200, showlegend=False)
        return fig
    
    def save_model(self, directory='models'):
        """Save model and scaler"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(directory, f'pattern_predictor_{timestamp}')
        
        # Save components
        joblib.dump({
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
        }, f'{model_path}.joblib')
        
        print(f"Model saved to {model_path}.joblib")
        
    @classmethod
    def load_model(cls, model_path):
        """Load saved model"""
        components = joblib.load(model_path)
        
        predictor = cls()
        predictor.classifier = components['classifier']
        predictor.regressor = components['regressor']
        predictor.scaler = components['scaler']
        
        return predictor