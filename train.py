from model.Net import Net
import csv
import numpy as np

def one_hot_encode(labels, num_classes):
    """
    Convert labels to one-hot encoded format.
    labels: Array of integer labels
    num_classes: Total number of classes
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def load_data(file_path):
    """
    Load data from a CSV file and return features and labels as numpy arrays.
    """

    features = []
    labels = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            features.append([float(x) for x in row[1:]])  # All columns except the first one are features
            labels.append(int(row[0]))  # The first column is the label
    return np.array(features), np.array(labels)

file_path = './data/train.csv'

# Load the training data
X_train, y_train = load_data(file_path)
y_train_one_hot = one_hot_encode(y_train, num_classes=10)  # Assuming 10 classes
X_train = X_train / 255.0  # Normalize pixel values to [0, 1]

# Define the neural network architecture
net = Net([784, 128, 64, 10])

# Training parameters
learning_rate = 0.01
batch_size = 64
epochs = 30
num_samples = X_train.shape[0]
save_path = './model_params.npz'  # Where to export the trained model parameters

def compute_accuracy(net, X, Y_one_hot):
    """
    Compute the accuracy of predictions against true labels.
    predictions: Array of predicted class indices
    labels: Array of true class indices
    """

    logits = net.forward(X)
    predictions = np.argmax(logits, axis=1)
    true_labels = np.argmax(Y_one_hot, axis=1)
    return np.mean(predictions == true_labels)

for epoch in range(epochs):
    perm = np.random.permutation(num_samples)
    X_shuffled = X_train[perm]
    y_shuffled = y_train_one_hot[perm]

    total_loss = 0.0
    num_batches = 0

    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # Forward pass
        logits = net.forward(X_batch)
        loss = net.loss_fn.forward(logits, y_batch)
        total_loss += loss
        num_batches += 1

        # Backward pass
        dLoss = net.loss_fn.backward()
        net.backward(dLoss)

        # Update parameters
        net.step(learning_rate)

    avg_loss = total_loss / num_batches

    train_acc = compute_accuracy(net, X_train, y_train_one_hot)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

# Export the trained model parameters for use in external programs
net.save(save_path)
print(f"Model parameters saved to {save_path}")