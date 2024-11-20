import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 加载数据
data = pd.read_csv('original data.csv')
X = data.iloc[:, 0:54].values
y = data.iloc[:, 55:56].values

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 定义模型
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.layer1 = nn.Linear(X.shape[1], 108)  # 增加神经元数量
        self.bn1 = nn.BatchNorm1d(108)
        self.layer2 = nn.Linear(108, 216)
        self.bn2 = nn.BatchNorm1d(216)
        self.layer3 = nn.Linear(216, 216)
        self.bn3 = nn.BatchNorm1d(216)
        self.layer4 = nn.Linear(216, 1)
        self.dropout = nn.Dropout(0.2)  # Dropout比例调整为0.5

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.layer4(x)
        return x


model = DNNModel()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # 调整正则化权重

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

# 早停法参数
early_stopping_patience = 50
best_test_loss = float('inf')
trigger_times = 0

# 训练模型
num_epochs = 2000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

    scheduler.step(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        trigger_times = 0
    else:
        trigger_times += 1

    if trigger_times >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 计算R方
model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)
    train_r2 = 1 - ((train_pred - y_train) ** 2).sum() / ((y_train - y_train.mean()) ** 2).sum()
    test_r2 = 1 - ((test_pred - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()

print(f'Train R²: {train_r2:.4f}')
print(f'Test R²: {test_r2:.4f}')