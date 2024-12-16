import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer


"""
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Model                           | Pros                                                        | Cons                                                        |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| LSTM (Long Short-Term Memory)   | - Handles long-range dependencies well.                     | - Computationally expensive for long sequences.             |
|                                 | - Mitigates vanishing gradient problem.                     | - Slower training due to more parameters.                   |
|                                 | - Suitable for complex sequences (e.g., time-series, text). |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| GRU (Gated Recurrent Unit)      | - Simpler architecture (fewer parameters).                  | - Struggles with very long sequences compared to LSTM.      |
|                                 | - Faster training than LSTM, often similar performance.     | - Fewer gates may limit flexibility for complex tasks.      |
|                                 | - Effective at learning temporal dependencies.              |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| RNN (Recurrent Neural Network)  | - Simple and easy to implement.                             | - Struggles with long-range dependencies(vanishing gradient)|
|                                 | - Good for short-term dependencies.                         | - Not suitable for complex tasks without enhancements.      |
|                                 | - Lower computational cost than LSTM/GRU.                   |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| BiLSTM (Bidirectional LSTM)     | - Captures both past and future context.                    | - More computationally expensive than unidirectional LSTM.  |
|                                 | - Suitable for tasks where future context is important.     | - Increases train time due to processing in both directions.|
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| 1D CNN (1D Convolutional        | - Efficient for local patterns in sequential data.          | - Limited at capturing long-term deps comparing to RNNs.    |
| Neural Networks)                |                                                             |                                                             |
|                                 | - Faster than RNNs for certain cases.                       | - Requires tuning of kernel size for optimal performance.   |
|                                 | - Can handle high-dimensional inputs.                       |                                                             |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+
| Attention Mechanisms            | - Highly parallelizable, faster training.                   | - Requires large amounts of data for effective training.    |
| (e.g., Transformer)             |                                                             |                                                             |
|                                 | - Great at capturing long-range dependencies.               | - Computationally intensive with long sequences.            |
|                                 | - Scalable to large datasets and sequences.                 | - Complexity in hyperparameter tuning.                      |
+---------------------------------+-------------------------------------------------------------+-------------------------------------------------------------+

+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Normalization Type     | Formula                                          | Best Used For                              | Characteristics                                  |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Z-Score Normalization  | (x - μ) / σ                                      | - Normally distributed data                | - Mean = 0                                       |
| (StandardScaler)       | Transforms to mean 0, std dev 1                  | - Features on similar scales               | - Std Dev = 1                                    |
|                        |                                                  | - When outliers are not extreme            | - Sensitive to outliers                          |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Min-Max Scaling        | (x - min(x)) / (max(x) - min(x))                 | - When you need bounded range              | - Preserves zero values                          |
| (MinMaxScaler)         | Scales to fixed range (default [0,1])            | - Neural network inputs                    | - Sensitive to outliers                          |
|                        |                                                  | - Image processing                         | - Does not handle new min/max in test data       |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Robust Scaling         | (x - median) / IQR                               | - Data with many outliers                  | - Uses median instead of mean                    |
| (RobustScaler)         | Uses median and interquartile range              | - Skewed distributions                     | - Less affected by extreme values                |
|                        |                                                  | - Financial data with extreme values       | - Preserves shape of distribution                |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| MaxAbs Scaling         | x / max(abs(x))                                  | - Sparse data                              | - Preserves sign of original values              |
| (MaxAbsScaler)         | Scales by maximum absolute value                 | - When zero is meaningful                  | - Does not center data                           |
|                        |                                                  | - Machine learning with sparse features    | - Bounded between -1 and 1                       |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Quantile Transformation| Transforms to uniform/normal distribution        | - Non-gaussian distributions               | - Makes data look like normal distribution       |
| (QuantileTransformer)  | Equalizes feature distributions                  | - When feature distributions differ        | - Can handle non-linear transformations          |
|                        |                                                  | - Machine learning with varied features    | - Destroys sparseness                            |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
| Power Transformation   | x^λ or log(x)                                    | - Right-skewed data                        | - Stabilizes variance                            |
| (PowerTransformer)     | Stabilizes variance and makes data more Gaussian | - Financial ratios                         | - Handles positive values                        |
|                        |                                                  | - Economic indicators                      | - Different methods like Yeo-Johnson, Box-Cox    |
+------------------------+--------------------------------------------------+--------------------------------------------+--------------------------------------------------+
"""

class PatternRecognizer:
    def __init__(self, 
                 sequence_length=10, 
                 shape_features=2, 
                 additional_features=5,
                 price_normalization='standard',
                 additional_features_normalization='standard',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize multi-feature pattern recognition model with PyTorch
        
        Args:
            sequence_length (int): Number of time steps
            shape_features (int): Number of price-related features
            additional_features (int): Number of extra features
            price_normalization (str): Normalization method
            additional_features_normalization (str): Normalization method
            device (str): Computing device (cuda/cpu)
        """
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.shape_features = shape_features
        self.additional_features = additional_features
        
        # Select normalization scalers
        self.price_scaler = self._get_scaler(price_normalization)
        self.additional_features_scaler = self._get_scaler(additional_features_normalization)
        
        # Model architecture
        self.model = self._create_bilstm_model().to(self.device)
        
        # Compiled (optimized) model will be stored here
        self.compiled_model = None

    def _get_scaler(self, method):
        """
        Return appropriate sklearn scaler based on method name
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        if method.lower() not in scalers:
            raise ValueError(f"Unsupported normalization method: {method}. "
                             f"Choose from {list(scalers.keys())}")
        
        return scalers[method.lower()]

    def _create_bilstm_model(self):
        """
        Create PyTorch BiLSTM model
        """
        class BiLSTMModel(nn.Module):
            def __init__(self, sequence_length, shape_features, additional_features):
                super().__init__()
                # BiLSTM Layer
                self.bilstm = nn.LSTM(
                    input_size=shape_features, 
                    hidden_size=64, 
                    batch_first=True, 
                    bidirectional=True
                )
                
                # Fully connected layers
                self.fc_layers = nn.Sequential(
                    nn.Linear(128 + additional_features, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )

            def forward(self, price_sequence, additional_features):
                # BiLSTM processing
                lstm_out, _ = self.bilstm(price_sequence)
                
                # Take the last output of BiLSTM
                lstm_out = torch.cat([lstm_out[:, -1, :64], lstm_out[:, 0, 64:]], dim=1)
                
                # Combine LSTM output with additional features
                combined = torch.cat([lstm_out, additional_features], dim=1)
                
                # Pass through fully connected layers
                return self.fc_layers(combined)

        return BiLSTMModel(
            self.sequence_length, 
            self.shape_features, 
            self.additional_features
        )

    def preprocess_data(self, price_sequences, additional_features):
        """
        Preprocess and normalize input data
        """
        # Normalize price sequences
        price_sequences_normalized = self.price_scaler.fit_transform(
            price_sequences.reshape(-1, self.shape_features)
        ).reshape(-1, self.sequence_length, self.shape_features)
        
        # Normalize additional features
        additional_features_normalized = self.additional_features_scaler.fit_transform(
            additional_features
        )
        
        # Convert to PyTorch tensors
        price_tensor = torch.FloatTensor(price_sequences_normalized).to(self.device)
        features_tensor = torch.FloatTensor(additional_features_normalized).to(self.device)
        
        return price_tensor, features_tensor

    def train(self, price_sequences, additional_features, labels, epochs=50, batch_size=32):
        """
        Train the PyTorch BiLSTM model
        """
        # Preprocess data
        price_tensor, features_tensor = self.preprocess_data(
            price_sequences, additional_features
        )
        labels_tensor = torch.FloatTensor(labels).to(self.device)

        # Create DataLoader
        dataset = TensorDataset(price_tensor, features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and Optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for price_batch, features_batch, labels_batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(price_batch, features_batch).squeeze()
                loss = criterion(outputs, labels_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')

    def compile_model(self):
        """
        Create a TorchScript compiled version of the model for fastest inference
        """
        # Example input sizes
        example_price_sequence = torch.randn(1, self.sequence_length, self.shape_features).to(self.device)
        example_additional_features = torch.randn(1, self.additional_features).to(self.device)
        
        # Compile the model using TorchScript
        self.compiled_model = torch.jit.trace(
            self.model, 
            (example_price_sequence, example_additional_features)
        )
        
        print("Model compiled successfully for optimized inference")

    def predict(self, price_sequences, additional_features):
        """
        Make predictions using the model
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Preprocess data
        price_tensor, features_tensor = self.preprocess_data(
            price_sequences, additional_features
        )
        
        # Predict
        with torch.no_grad():
            return self.model(price_tensor, features_tensor).cpu().numpy()

    def predict_single_sample(self, price_sequence, additional_features):
        """
        Make predictions for a single sample with optimized inference
        """
        # Use compiled model if available, otherwise use regular model
        inference_model = self.compiled_model if self.compiled_model is not None else self.model
        
        # Ensure model is in evaluation mode
        inference_model.eval() # type: ignore
        
        # Normalize price sequence
        price_sequence_normalized = self.price_scaler.fit_transform(
            price_sequence.reshape(-1, self.shape_features)
        ).reshape(1, self.sequence_length, self.shape_features)
        
        # Normalize additional features
        additional_features_normalized = self.additional_features_scaler.fit_transform(
            additional_features.reshape(1, -1)
        )
        
        # Convert to PyTorch tensors
        price_tensor = torch.FloatTensor(price_sequence_normalized).to(self.device)
        features_tensor = torch.FloatTensor(additional_features_normalized).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = inference_model(price_tensor, features_tensor) # type: ignore
        
        return prediction.cpu().numpy()[0][0]

    def save_model(self, filename):
        """
        Save the entire model state
        """
        state = {
            'model_state': self.model.state_dict(),
            'price_scaler': self.price_scaler,
            'additional_features_scaler': self.additional_features_scaler
        }
        torch.save(state, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load the entire model state
        """
        state = torch.load(filename)
        self.model.load_state_dict(state['model_state'])
        self.price_scaler = state['price_scaler']
        self.additional_features_scaler = state['additional_features_scaler']
        self.model.eval()  # Set to evaluation mode
        print(f"Model loaded from {filename}")

def main():
    # Initialize and train model
    recognizer = PatternRecognizer(
        price_normalization='robust',
        additional_features_normalization='quantile'
    )
    
    # Generate synthetic data
    train_sequences, train_features, train_labels = _generate_synthetic_data()
    
    # Train the model
    recognizer.train(train_sequences, train_features, train_labels)
    
    # Compile model for fastest inference
    recognizer.compile_model()
    
    # Make predictions
    predictions = recognizer.predict(train_sequences, train_features)
    
    # Print results
    for i, (pred, true_label) in enumerate(zip(predictions, train_labels)):
        print(f"Sample {i}: Prediction={pred[0]:.2f}, True Label={true_label}")

def _generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic data (similar to original implementation)
    """
    np.random.seed(42)
    
    price_sequences = []
    additional_features_list = []
    labels = []
    
    for _ in range(num_samples):
        if np.random.rand() > 0.9:
            # Head and Shoulders pattern
            head_shoulders = np.array([
                [1, 2 + np.random.normal(0, 0.2)], 
                [2, 3 + np.random.normal(0, 0.2)], 
                [3, 2 + np.random.normal(0, 0.2)], 
                [4, 4 + np.random.normal(0, 0.2)], 
                [5, 2 + np.random.normal(0, 0.2)], 
                [6, 3 + np.random.normal(0, 0.2)], 
                [7, 1 + np.random.normal(0, 0.2)], 
                [8, 2 + np.random.normal(0, 0.2)], 
                [9, 1 + np.random.normal(0, 0.2)], 
                [10, 2 + np.random.normal(0, 0.2)]
            ])
            label = 1
        else:
            # Random pattern
            head_shoulders = np.random.rand(10, 2) * 5 + np.random.normal(0, 0.2, (10, 2))
            label = 0
        
        # Generate additional features
        additional_features = np.random.rand(5)
        additional_features[0] = np.mean(head_shoulders[:, 1])  # Average price
        additional_features[1] = np.std(head_shoulders[:, 1])   # Price volatility
        additional_features[2] = np.max(head_shoulders[:, 1]) - np.min(head_shoulders[:, 1])  # Price range
        additional_features[3] = np.sum(np.diff(head_shoulders[:, 1]) > 0)  # Upward movements
        additional_features[4] = label * np.random.rand()  # Correlated noise
        
        price_sequences.append(head_shoulders)
        additional_features_list.append(additional_features)
        labels.append(label)
    
    return (
        np.array(price_sequences), 
        np.array(additional_features_list), 
        np.array(labels)
    )

if __name__ == "__main__":
    main()