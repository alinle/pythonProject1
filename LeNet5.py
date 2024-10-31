import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 3채널 입력으로 수정
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Flatten 후 맞는 크기로 수정
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc3 = nn.Linear(84, 58)  # OUT features은 class 수에 맞게 조절해야한다.

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc1(x)
        x = self.tanh3(x)
        x = self.fc2(x)
        x = self.tanh4(x)
        x = self.fc3(x)
        return x

