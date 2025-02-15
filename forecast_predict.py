import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
dir_path = os.path.dirname(os.path.realpath(__file__))
hdf5_file_path = dir_path + '/test.hdf5'
model_path = dir_path + '/saved_models_forecast/arm_transformer_final.pth'
thr = 0.15

# Define model and positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ArmTransformerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size, seq_length, dropout=0.1):
        super(ArmTransformerPredictor, self).__init__()

        self.input_size = input_size
        if self.input_size % num_heads != 0:
            raise ValueError(f"input_size ({self.input_size}) must be divisible by num_heads ({num_heads})")

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(self.input_size, output_size)
        self.pos_encoder = PositionalEncoding(self.input_size, dropout, max_len=seq_length)

    def forward(self, x):
        x_with_pos = self.pos_encoder(x)
        output = self.transformer_encoder(x_with_pos)
        output = self.fc(output)
        return output

# Load model and set it to evaluation mode
model = ArmTransformerPredictor(input_size=48, hidden_size=128, num_heads=12, output_size=1, seq_length=100).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to load data
def process_data(hdf5_file_path):
    inputs = []
    targets = []

    with h5py.File(hdf5_file_path, 'r') as hdf:
        for episode_name in hdf.keys():
            episode_group = hdf[episode_name]
            input_data = episode_group['input_data'][:]
            target_data = episode_group['target_data'][:]
            target_data = target_data[::10]

            inputs.append(input_data)
            targets.append(target_data)

    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)
    return inputs, targets

# Load test data
inputs, targets = process_data(hdf5_file_path)
dataset = TensorDataset(inputs, targets)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Prediction and Error Calculation
total_errors = []
last_element_errors = []
start_points = []
end_points_actual = []
end_points_predicted = []
end_points = []
total_seq = []

with torch.no_grad():
    for batch_inputs, batch_targets in test_dataloader:
        batch_size = batch_targets.size(0)
        batch_targets = batch_targets.view(batch_size, -1, 1)


        if batch_inputs.dim() == 2:
            batch_inputs = batch_inputs.unsqueeze(1)
        
        batch_inputs = batch_inputs.expand(batch_size, 100, -1)
        padding = torch.zeros(batch_size, 100, 5).to(batch_inputs.device)
        batch_inputs = torch.cat((batch_inputs, padding), dim=2)

        predictions = model(batch_inputs).view(batch_size, -1)

        # Calculate errors
        errors = (abs(predictions - batch_targets.squeeze(-1))).cpu().numpy()
        last_element_errors.extend(errors[:, -1])
        total_errors.extend(errors.flatten())
        seq = (batch_targets.squeeze(-1)).cpu().numpy()
        start_points.extend(seq[:,0])
        total_seq.extend(batch_targets.squeeze(-1).cpu().numpy())
        end_points_predicted.extend(predictions[:, -1].cpu().numpy())
        end_points_actual.extend(seq[:,-1])

# Compute statistics
mean_error = np.mean(total_errors)
std_error = np.std(total_errors)
ci_total_error = 1.96 * (std_error / np.sqrt(len(total_errors)))
last_mean_error = np.mean(last_element_errors)
last_std_error = np.std(last_element_errors)
ci_last_error = 1.96 * (last_std_error / np.sqrt(len(last_element_errors)))


print(f"Overall Error - Mean: {mean_error:.4f}, Std Dev: {std_error:.4f}")
print(f"Last Element Error - Mean: {last_mean_error:.4f}, Std Dev: {last_std_error:.4f}")

# Calculate the mean starting and ending points of the test data
mean_start_point = np.mean(start_points)
mean_end_point = np.mean(end_points_actual)

print(f"Mean Starting Point of Test Data: {mean_start_point:.4f}")
print(f"Mean Ending Point of Test Data: {mean_end_point:.4f}")

print(type(end_points_predicted))
print(type(end_points_actual))
end_points_predicted = np.array(end_points_predicted)
end_points_actual = np.array(end_points_actual)
print(len(end_points_actual))
sum_less_than_0_1 = (end_points_actual < 0.1).sum()


print("Sum of elements less than 0.1:", sum_less_than_0_1)

# Define categories based on the given conditions
predicted_categories = np.where((end_points_predicted > 0) & (end_points_predicted < thr), 1, 2)
actual_categories = np.where((end_points_actual > 0) & (end_points_actual < thr), 1, 2)

# Compute confusion matrix
conf_matrix = confusion_matrix(actual_categories, predicted_categories, labels=[1, 2])

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


true_negative = conf_matrix[1, 1]
false_positive = conf_matrix[0, 1]
false_negative = conf_matrix[1, 0]
true_positive = conf_matrix[0, 0]

# Calculate metrics
accuracy = (true_positive + true_negative) / np.sum(conf_matrix)
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

total_errors = []
last_element_errors = []
start_points = []
end_points_actual = []
end_points_predicted = []
end_points = []

with torch.no_grad():
    for batch_inputs, batch_targets in test_dataloader:
        batch_size = batch_targets.size(0)
        batch_targets = batch_targets.view(batch_size, -1, 1)


        if batch_inputs.dim() == 2:
            batch_inputs = batch_inputs.unsqueeze(1)
        
        batch_inputs = batch_inputs.expand(batch_size, 100, -1)
        padding = torch.zeros(batch_size, 100, 5).to(batch_inputs.device)
        batch_inputs = torch.cat((batch_inputs, padding), dim=2)

        predictions = model(batch_inputs).view(batch_size, -1)

        # Calculate errors
        errors = (abs(predictions - batch_targets.squeeze(-1))).cpu().numpy()
        last_element_errors.extend(errors[:, -1])
        total_errors.extend(errors.flatten())
        seq = (batch_targets.squeeze(-1)).cpu().numpy()
        start_points.extend(seq[:,0])
        
        end_points_predicted.extend(predictions[:, -1].cpu().numpy())
        end_points_actual.extend(seq[:,-1])

# Compute statistics
mean_error = np.mean(total_errors)
std_error = np.std(total_errors)
ci_total_error = 1.96 * (std_error / np.sqrt(len(total_errors)))
last_mean_error = np.mean(last_element_errors)
last_std_error = np.std(last_element_errors)
ci_last_error = 1.96 * (last_std_error / np.sqrt(len(last_element_errors)))


print(f"Overall Error - Mean: {mean_error:.4f}, Std Dev: {std_error:.4f}")
print(f"Last Element Error - Mean: {last_mean_error:.4f}, Std Dev: {last_std_error:.4f}")

# Calculate the mean starting and ending points of the test data
mean_start_point = np.mean(start_points)
mean_end_point = np.mean(end_points_actual)

print(f"Mean Starting Point of Test Data: {mean_start_point:.4f}")
print(f"Mean Ending Point of Test Data: {mean_end_point:.4f}")

print(type(end_points_predicted))
print(type(end_points_actual))
end_points_predicted = np.array(end_points_predicted)
end_points_actual = np.array(end_points_actual)
print(len(end_points_actual))
sum_less_than_0_1 = (end_points_actual < 0.1).sum()


print("Sum of elements less than 0.1:", sum_less_than_0_1)

# Define categories based on the given conditions
predicted_categories = np.where((end_points_predicted > 0) & (end_points_predicted < thr), 1, 2)
actual_categories = np.where((end_points_actual > 0) & (end_points_actual < thr), 1, 2)

# Compute confusion matrix
conf_matrix = confusion_matrix(actual_categories, predicted_categories, labels=[1, 2])

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


true_negative = conf_matrix[1, 1]
false_positive = conf_matrix[0, 1]
false_negative = conf_matrix[1, 0]
true_positive = conf_matrix[0, 0]

# Calculate metrics
accuracy = (true_positive + true_negative) / np.sum(conf_matrix)
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

total_seq_array = np.array(total_seq)
total_errors_array = np.array(total_errors)


# Calculate the mean and uncertainty bounds
mean_total_seq = np.mean(total_seq_array, axis=0)
std_total_errors = np.std(total_errors_array, axis=0)

# Define the upper and lower bounds (mean ± standard deviation)
upper_bound = mean_total_seq + std_total_errors
lower_bound = mean_total_seq - std_total_errors

# Plot the mean with the uncertainty band
plt.figure(figsize=(10, 6))
plt.plot(mean_total_seq, label="Mean Total Sequence", color="blue")
plt.fill_between(range(len(mean_total_seq)), lower_bound, upper_bound, color="blue", alpha=0.2, label="Uncertainty Band")

# Add labels and title
plt.xlabel("Time Steps")
plt.ylabel("Sequence Values")
plt.title("Mean Total Sequence with Uncertainty Band")
plt.legend()
plt.show()

#Define parameters
sequence_length = 100
num_sequences = len(actual_categories)  # This should be the number of unique sequences

# Reshape total_seq and total_errors to have shape (num_sequences, sequence_length)
total_seq_array = np.array(total_seq).reshape(num_sequences, sequence_length)
total_errors_array = np.array(total_errors).reshape(num_sequences, sequence_length)

# Repeat each category 100 times to match sequence length
expanded_categories = np.repeat(actual_categories, sequence_length)

# Flatten the reshaped arrays to use with expanded categories
total_seq_flat = total_seq_array.flatten()
total_errors_flat = total_errors_array.flatten()

# Separate data by category using expanded categories
category_1_seq = total_seq_flat[expanded_categories == 1]
category_2_seq = total_seq_flat[expanded_categories == 2]

category_1_errors = total_errors_flat[expanded_categories == 1]
category_2_errors = total_errors_flat[expanded_categories == 2]

# Function to plot mean and uncertainty band for a given category
def plot_category_data(seq_data, errors_data, category_name):
    # Reshape to get sequences back for calculating mean and uncertainty band
    seq_data = seq_data.reshape(-1, sequence_length)
    errors_data = errors_data.reshape(-1, sequence_length)

    mean_seq = np.mean(seq_data, axis=0)  # Mean across sequences
    std_errors = np.std(errors_data, axis=0)  # Std deviation across sequences for uncertainty

    upper_bound = mean_seq + std_errors
    lower_bound = mean_seq - std_errors

    plt.plot(mean_seq, label=f"Mean Total Sequence - Category {category_name}", color="blue")
    plt.fill_between(range(sequence_length), lower_bound, upper_bound, color="blue", alpha=0.2, label="Uncertainty Band")
    plt.xlabel("Time Steps")
    plt.ylabel("Sequence Values")
    plt.title(f"Mean Total Sequence with Uncertainty Band - Category {category_name}")
    plt.legend()
    plt.show()

# Plot for Category 1
plt.figure(figsize=(10, 6))
plot_category_data(category_1_seq, category_1_errors, category_name="1")

# Plot for Category 2
plt.figure(figsize=(10, 6))
plot_category_data(category_2_seq, category_2_errors, category_name="2")