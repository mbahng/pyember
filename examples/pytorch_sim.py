import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Create a simple dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        # Generate random input features (x1, x2)
        self.x = torch.randn(size, 2)
        # Generate target: y = x1 + 2*x2 + noise
        self.y = self.x[:, 0] + 2 * self.x[:, 1] + torch.randn(size) * 0.1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x.squeeze()

# Create datasets and dataloaders
train_dataset = SimpleDataset(size=1000)
test_dataset = SimpleDataset(size=200)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'\nTest Loss: {avg_test_loss:.4f}')

# Make some predictions
test_samples = torch.tensor([
    [1.0, 0.5],
    [-0.5, 1.0],
    [0.0, -1.0]
])

model.eval()
with torch.no_grad():
    predictions = model(test_samples)
    print("\nTest Predictions:")
    for x, pred in zip(test_samples, predictions):
        # Expected value would be approximately x[0] + 2*x[1]
        expected = x[0].item() + 2 * x[1].item()
        print(f"Input: {x.numpy()}, Predicted: {pred:.4f}, Expected: {expected:.4f}")

