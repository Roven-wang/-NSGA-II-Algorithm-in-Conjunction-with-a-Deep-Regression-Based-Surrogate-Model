import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Dataset class for loading data
class MyData(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv('training data.csv', index_col=0)  # Load data
        arr = df.values.astype(np.float32)  # Convert to numpy array
        ts = torch.tensor(arr).to('cuda')  # Convert to tensor and move to GPU
        self.X = ts[:, :43]  # First 43 columns as features
        self.Y = ts[:, 43:44]  # 44th column as target
        self.len = ts.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len

# Split data into training and testing sets
Data = MyData('training data.csv')
train_size = int(len(Data) * 0.7)
test_size = len(Data) - train_size
torch.manual_seed(25)
np.random.seed(25)
train_Data, test_Data = random_split(Data, [train_size, test_size])

# DataLoader for batching
train_loader = DataLoader(dataset=train_Data, shuffle=True, batch_size=32)
test_Loader = DataLoader(dataset=test_Data, shuffle=False, batch_size=32)

# Define the neural network
class DNN1(nn.Module):
    def __init__(self):
        super(DNN1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(43, 100), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.net(x)

model = DNN1().to('cuda:0')

# Loss function and optimizer
loss_fn = nn.MSELoss()
learning_rate = 0.0001
weight_decay = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the model
epochs = 200
train_losses = []
val_losses = []
for epoch in range(epochs):
    epoch_train_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        Pred = model(x)
        loss = loss_fn(Pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in test_Loader:
            Pred_val = model(x_val)
            val_loss = loss_fn(Pred_val, y_val)
            running_val_loss += val_loss.item()
    val_loss = running_val_loss / len(test_Loader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch + 1}/{epochs}], Average Training Loss: {epoch_train_loss:.4f} , Validation Loss: {val_loss:.4f}')

# Plot training and validation losses
'''plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()'''

# Save training and validation losses to Excel
'''loss_data = pd.DataFrame({
    'Epoch': range(1, epochs + 1),
    'Training Loss': train_losses,
    'Validation Loss': val_losses
})
loss_data.to_excel('loss_data.xlsx', index=False)'''

# Define a test function
'''def test(model, test_Loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_Loader:
            pred = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return y_true, y_pred, r2, rmse, mae

# Evaluate on training set
y_train_true, y_train_pred, r2, rmse, mae = test(model, train_loader)
print(f'Training R²: {r2}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# Evaluate on testing set
y_test_true, y_test_pred, r2, rmse, mae = test(model, test_Loader)
print(f'Testing R²: {r2}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# Visualize predictions
def visualize_results(y_true, y_pred, title):
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

visualize_results(y_train_true, y_train_pred, 'Actual vs Predicted (Training Set)')
visualize_results(y_test_true, y_test_pred, 'Actual vs Predicted (Testing Set)')'''

# Save actual and predicted values to Excel
'''train_results = pd.DataFrame({
    'Actual': y_train_true,
    'Predicted': y_train_pred
})
test_results = pd.DataFrame({
    'Actual': y_test_true,
    'Predicted': y_test_pred
})
with pd.ExcelWriter( r'F:\desktop\results_data.xlsx') as writer:
    train_results.to_excel(writer, sheet_name='Training Set', index=False)
    test_results.to_excel(writer, sheet_name='Testing Set', index=False)'''

# Save the model
torch.save(model.state_dict(), r'H:\python\54parameter\DNN-NSE.pth')
