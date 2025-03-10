import array
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Constants for the sliding windows.
LEN_F = 10  # Number of past data points used as features.
LEN_P = 5   # Number of data points used to compute the label.

class LSTM2Layer(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        # Use the output from the last time step.
        return self.fc(out[:, -1, :])

class LSTM:
    def __init__(self,
                 strength: array.array,
                 tr_mult: array.array,
                 v_mult: array.array,
                 atr: array.array,
                 rsi: array.array,
                 log_returns: array.array):
        # The following arrays are assumed to be mutable and updated externally.
        self.strength = strength
        self.tr_mult = tr_mult
        self.v_mult = v_mult
        self.atr = atr
        self.rsi = rsi
        self.log_returns = log_returns

        self.data_points = 0 # Count of how many times update() has been called.
        self.label = array.array('d', []) # Store labels as plain Python lists.
        self.prediction = array.array('d', []) # Store predictions as plain Python lists.

        self.model = LSTM2Layer(input_size=6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model.train()  # Start in training mode.

    def update(self):
        """
        Called whenever new data is appended to the mutable arrays.
        Performs:
          1. On-the-fly prediction using the latest available window.
          2. Every 100 updates, extracts the last 100 training samples and trains in batch.
          3. Prints current sign accuracy over the saved recent samples.
        """
        self.data_points += 1
        total = len(self.log_returns)
        # Need at least LEN_F + LEN_P data points to form one sample.
        if total < (LEN_F + LEN_P):
            self.label.append(self.log_returns[-1])
            self.prediction.append(self.log_returns[-1])
            return self.log_returns[-1]

        # Convert the mutable arrays to NumPy arrays once.
        arrays = [np.asarray(arr) for arr in 
                  (self.strength, self.tr_mult, self.v_mult, self.atr, self.rsi, self.log_returns)]

        # --- On-the-Fly Prediction ---
        # Extract a window of LEN_F data points (features) immediately preceding the label window.
        features = np.column_stack([arr[-(LEN_F + LEN_P):-LEN_P] for arr in arrays])
        # Label: average of the last LEN_P log_returns.
        label = np.mean(arrays[-1][-LEN_P:])
        self.label.append(float(label))

        # Run prediction (batch size = 1).
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Shape: (1, LEN_F, 6)
        self.model.eval()  # Inference mode.
        with torch.no_grad():
            prediction = self.model(input_tensor).item()
        self.prediction.append(prediction)

        # --- Print Current Sign Accuracy ---
        # Check accuracy over the most recent 100 samples (or fewer if not available).
        recent_count = min(len(self.label), 100)
        if recent_count > 0:
            recent_labels = np.array(self.label[-recent_count:])
            recent_preds = np.array(self.prediction[-recent_count:])
            # Compare signs: a prediction is "correct" if its sign matches the label's sign.
            correct = np.sum(np.sign(recent_labels) == np.sign(recent_preds))
            accuracy = correct / recent_count * 100
            print(f"Current sign accuracy over last {recent_count} samples: {accuracy:.2f}%")

        # --- Batch Training Every 100 Updates ---
        # Only train if we have enough data.
        if self.data_points % 100 == 0 and total >= (LEN_F + LEN_P + 100 - 1):
            self.train_batch(arrays, total)

        # Optionally trim the history lists to avoid uncontrolled growth.
        MAX_HISTORY = 100
        if len(self.prediction) > 2 * MAX_HISTORY:
            self.prediction = self.prediction[-MAX_HISTORY:]
        if len(self.label) > 2 * MAX_HISTORY:
            self.label = self.label[-MAX_HISTORY:]

        return prediction

    def train_batch(self, arrays, total):
        """
        Extract the last 100 training samples from the mutable arrays and perform a batch training step.
        Each sample uses:
          - Features: a window of LEN_F data points.
          - Label: the average of the following LEN_P log_returns.
        """
        batch_size = 100
        inputs, labels = [], []
        # For each training sample, let i be such that:
        #    features: indices [i - (LEN_F+LEN_P), i - LEN_P)
        #    label: average over indices [i - LEN_P, i)
        # Loop over indices that yield 100 samples.
        start_indices = range(total - batch_size + 1 - (LEN_F + LEN_P), total - (LEN_F + LEN_P) + 1)
        for s in start_indices:
            feat = np.column_stack([arr[s:s + LEN_F] for arr in arrays])
            lab = np.mean(arrays[-1][s + LEN_F:s + LEN_F + LEN_P])
            inputs.append(feat)
            labels.append(lab)

        # Convert the lists to torch tensors.
        batch_inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        batch_labels = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)

        # Train the model on this batch.
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(batch_inputs)
        loss = self.criterion(outputs, batch_labels)
        loss.backward()
        self.optimizer.step()
        # Optionally, log the loss:
        # print(f"Batch training loss: {loss.item():.5f}")
