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
        self.Y = ts[:, 44:45]  # 44th column as target
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
train_loader = DataLoader(dataset=train_Data, shuffle=True, batch_size=16)
test_Loader = DataLoader(dataset=test_Data, shuffle=False, batch_size=16)

# Define the neural network
class DNN2(nn.Module):
    def __init__(self):
        '''搭建神经网络各层'''
        super(DNN2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(43, 80), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(80, 80), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(80, 1)
        )

    def forward(self,x):
        '''前向传播'''
        y = self.net(x)
        return y

model = DNN2().to('cuda:0')
#print(model)

# Loss function and optimizer
loss_fn = nn.MSELoss()
learning_rate = 0.0002 #设置学习率
weight_decay = 0.0001 #设置正则化参数
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
loss_data.to_excel('loss_data2.xlsx', index=False)'''

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
with pd.ExcelWriter( r'F:\desktop\results_data2.xlsx') as writer:
    train_results.to_excel(writer, sheet_name='Training Set', index=False)
    test_results.to_excel(writer, sheet_name='Testing Set', index=False)'''

# Save the model
torch.save(model.state_dict(), r'H:\python\54parameter\DNN-BR2.pth')




# 加载 Excel 文件
'''file_path = r'F:\desktop\测试.xlsx'
data = pd.read_excel(file_path)

# 将数据转换为 PyTorch 张量，并移动到 GPU（如果模型在 GPU 上）
input_tensor = torch.tensor(data.values.astype(np.float32))
input_tensor = input_tensor.to('cuda')

# 使用模型进行预测
model.eval()
with torch.no_grad():
    predicted_output = model(input_tensor)

# 将预测结果转换为 NumPy 数组
predicted_output = predicted_output.cpu().numpy()

# 打印每次模拟的预测结果
for i, pred in enumerate(predicted_output):
    print(f"Simulation {i+1}: {pred}")'''