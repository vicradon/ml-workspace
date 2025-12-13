import torch
import torch.nn as nn
import numpy as np

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(100, 1) * 10
y = 2 * X + 1 + torch.randn(100, 1) * 2  # y = 2x + 1 + noise

# Linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Get trained parameters
weight = model.linear.weight.item()
bias = model.linear.bias.item()
print(f'Trained parameters: weight = {weight:.4f}, bias = {bias:.4f}')
print(f'Expected: weight = 2.0000, bias = 1.0000')