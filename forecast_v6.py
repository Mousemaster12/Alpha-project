import os
import torch
import h5py
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np
import math

# TRANSFORMER

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the HDF5 file
dir_path = os.path.dirname(os.path.realpath(__file__))
hdf5_file_path = dir_path + '/training_forecast/training_data.hdf5'

model_dir = dir_path + '/saved_models_forecast'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Function to process the data from HDF5 file
def process_data(hdf5_file_path):
    inputs = []
    targets = []
    
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as hdf:
        # Iterate through each episode in the HDF5 file
        for episode_name in hdf.keys():
            episode_group = hdf[episode_name]
            
            # Read input data and target data for the current episode
            input_data = episode_group['input_data'][:]  # Input is already concatenated
            target_data = episode_group['target_data'][:]  # Distance sequence

            # Take every 10th element from the target data
            target_data = target_data[::10]  # Slicing to get every 10th element
            
            # Append the input and target to the lists
            inputs.append(input_data)
            targets.append(target_data)
    
    # Convert lists to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    return inputs, targets


# Call the function to get inputs and targets
inputs, targets = process_data(hdf5_file_path)
print(f"Inputs shape: {inputs.shape}")
print(f"Targets shape: {targets.shape}")

# Parameters
input_size = 30 + 7 + 3 + 3  # 43 total input features
seq_length = 100  # Sequence length of the output vector (distances)
batch_size = 32
hidden_size = 128
num_layers = 2
output_size = 100  # Distance at each time step (1D)

# Check that input and target shapes match expected format
assert inputs.shape[1] == input_size, f"Expected input size of {input_size}, but got {inputs.shape[1]}"
assert targets.shape[1] == seq_length, f"Expected target sequence length of {seq_length}, but got {targets.shape[1]}"

# Create a dataset
dataset = TensorDataset(inputs, targets)

# Split the dataset into train (90%) and test (10%)
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for both training and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#########################################################################################################
########################################### TRAINING ####################################################
#########################################################################################################

# Create a weight vector that emphasizes the beginning and end of the sequence
def create_weight_vector(seq_length):
    # Using a simple triangular weighting, where weights are higher at the start and end
    weights = np.ones(seq_length)
    mid = seq_length // 2
    for i in range(mid):
        weights[i] = 1 - (i / mid)  # Linearly decrease weights in the first half
        weights[seq_length - 1 - i] = 1 - (i / mid)  # Same for the second half
    
    return torch.tensor(weights, dtype=torch.float32).to(device)  # Move to device

# def create_weight_vector(seq_length):
#     # Create an exponential weight vector with more emphasis on the end of the sequence
#     weights = np.exp(np.linspace(0, 5, seq_length))  # Exponentially increasing weights
#     weights = weights / np.max(weights)  # Normalize so that the maximum weight is 1.0
#     return torch.tensor(weights, dtype=torch.float32).to(device) 

# Function to apply weighted loss
def weighted_mse_loss(predictions, targets, weights):
    # Apply the weights to both predictions and targets
    loss = (weights * (predictions - targets) ** 2).mean()
    return loss

# Define the LSTM-based neural network (same as before)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cosine to odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ArmTransformerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size, seq_length, dropout=0.1):
        super(ArmTransformerPredictor, self).__init__()

        self.input_size = input_size  # Keep the original input size

        # Ensure d_model is divisible by num_heads
        if self.input_size % num_heads != 0:
            raise ValueError(f"input_size ({self.input_size}) must be divisible by num_heads ({num_heads})")

        # Define the transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Fully connected layer for output
        self.fc = nn.Linear(self.input_size, output_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.input_size, dropout, max_len=seq_length)

    def forward(self, x):
        # Apply positional encoding
        x_with_pos = self.pos_encoder(x)

        # Pass through the Transformer encoder
        output = self.transformer_encoder(x_with_pos)

        # Pass through the fully connected layer
        output = self.fc(output)
        
        return output

# Instantiate the model with example parameters
model = ArmTransformerPredictor(
    input_size=48,    # Original input size
    hidden_size=128,  # Transformer feedforward layer size
    output_size=1,    # Output size (distance prediction)
    seq_length=100,   # Example sequence length
    num_heads=12       # Number of attention heads
).to(device)

# Create the weight vector for the loss function
weights = create_weight_vector(seq_length)

# Example training loop
num_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Flag to choose whether to use weighted loss or not
use_weights = False  # Set to False if you don't want to use weights

pad = 5 

# Modify the training loop to handle both weighted and unweighted loss
for epoch in range(num_epochs):
    model.train()

    for batch_inputs, batch_targets in train_dataloader:
        # Reshape targets as necessary
        current_batch_size = batch_targets.size(0)
        batch_targets = batch_targets.view(current_batch_size, seq_length, 1)
        
        # Ensure batch_inputs has 3 dimensions
        if batch_inputs.dim() == 2:  # If only [current_batch_size, feature_size]
            batch_inputs = batch_inputs.unsqueeze(1)  # Add a dimension: [current_batch_size, 1, feature_size]
        
        # Now repeat along the seq_length dimension to match expected shape
        batch_inputs = batch_inputs.expand(current_batch_size, seq_length, -1)  # Shape: [current_batch_size, seq_length, 46]

        # Create padding with zeros to expand input features
        padding = torch.zeros(current_batch_size, seq_length, pad).to(batch_inputs.device)  # shape (current_batch_size, seq_length, 2)

        # Concatenate padding to the original inputs
        batch_inputs = torch.cat((batch_inputs, padding), dim=2)  # shape becomes (current_batch_size, seq_length, 48)

        # Forward pass
        output = model(batch_inputs).view(current_batch_size, seq_length, 1)

        # Compute loss (weighted or unweighted)
        if use_weights:
            loss = weighted_mse_loss(output, batch_targets, weights)
        else:
            loss = nn.MSELoss()(output, batch_targets)  # Standard unweighted MSE loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # Save the model after every 10 epochs
        model_save_path = f'{model_dir}/arm_transformer_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved at {model_save_path}')

# Modify the test loop as well
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_inputs, batch_targets in test_dataloader:
            # Reshape targets as necessary
            current_batch_size = batch_targets.size(0)
            batch_targets = batch_targets.view(current_batch_size, seq_length, 1)
            
            # Ensure batch_inputs has 3 dimensions
            if batch_inputs.dim() == 2:  # If only [current_batch_size, feature_size]
                batch_inputs = batch_inputs.unsqueeze(1)  # Add a dimension: [current_batch_size, 1, feature_size]
            
            # Now repeat along the seq_length dimension to match expected shape
            batch_inputs = batch_inputs.expand(current_batch_size, seq_length, -1)  # Shape: [current_batch_size, seq_length, 46]

            # Create padding with zeros to expand input features
            padding = torch.zeros(current_batch_size, seq_length, pad).to(batch_inputs.device)  # shape (current_batch_size, seq_length, 2)

            # Concatenate padding to the original inputs
            batch_inputs = torch.cat((batch_inputs, padding), dim=2)  # shape becomes (current_batch_size, seq_length, 48)
            
            # Debugging output to check shapes
            print(f"batch_inputs: {batch_inputs.shape}")

            # Forward pass
            output = model(batch_inputs).view(current_batch_size, seq_length, 1)

            # Compute loss (weighted or unweighted)
            if use_weights:
                loss = weighted_mse_loss(output, batch_targets, weights)
            else:
                loss = nn.MSELoss()(output, batch_targets)

            test_loss += loss.item()

        # Calculate average test loss for the epoch
        test_loss /= len(test_dataloader)
        print(f'Test Loss: {test_loss:.4f}')

final_model_path = f'{model_dir}/arm_transformer_final.pth'
torch.save(model.state_dict(), final_model_path)
print(f'Final model saved at {final_model_path}')

#########################################################################################################
#######################################33 PLOTS TEST ####################################################
#########################################################################################################

# Function to plot predictions vs targets for random samples
def plot_predictions_vs_targets(model, dataloader, num_samples=4, seq_length=100):
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    with torch.no_grad():  # Disable gradient calculation for inference
        # Loop through the dataloader to extract predictions and targets
        for batch_inputs, batch_targets in dataloader:
            # Reshape targets as necessary
            current_batch_size = batch_targets.size(0)
            batch_targets = batch_targets.view(current_batch_size, seq_length, 1)

            # Ensure batch_inputs has 3 dimensions
            if batch_inputs.dim() == 2:  # If only [current_batch_size, feature_size]
                batch_inputs = batch_inputs.unsqueeze(1)  # Add a dimension: [current_batch_size, 1, feature_size]
            
            # Now repeat along the seq_length dimension to match expected shape
            batch_inputs = batch_inputs.expand(current_batch_size, seq_length, -1)  # Shape: [current_batch_size, seq_length, 46]

            # Create padding with zeros to expand input features
            padding = torch.zeros(current_batch_size, seq_length, pad).to(batch_inputs.device)  # shape (current_batch_size, seq_length, 2)

            # Concatenate padding to the original inputs
            batch_inputs = torch.cat((batch_inputs, padding), dim=2)  # shape becomes (current_batch_size, seq_length, 48)

            # Get predictions from the model
            batch_predictions = model(batch_inputs).view(current_batch_size, seq_length, 1)

            # Plot a few random samples
            for i in range(num_samples):
                # Choose a random sample from the batch
                idx = random.randint(0, current_batch_size - 1)

                # Get the actual and predicted sequences
                prediction = batch_predictions[idx].cpu().numpy()
                target = batch_targets[idx].cpu().numpy()

                # Create a plot
                plt.figure(figsize=(8, 4))
                plt.plot(prediction, label='Predicted')
                plt.plot(target, label='Actual', linestyle='dashed')
                plt.title(f'Sample {i+1}: Predicted vs Target Sequence')
                plt.xlabel('Time Step')
                plt.ylabel('Distance to Goal')
                plt.legend()
                plt.show()

            # Break after showing the specified number of samples
            break

# Call the function after training
plot_predictions_vs_targets(model, test_dataloader, num_samples=40)
