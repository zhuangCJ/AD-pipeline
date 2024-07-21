import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size=4096, hidden_size_1=2048, hidden_size_2=1024, hidden_size_3=512, hidden_size_4=128, num_classes=2):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, hidden_size_4)
        self.fc5 = nn.Linear(hidden_size_4, num_classes)
        self.relu = nn.ELU()
        self.drop_out = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

    def forward(self, x_1, x_2, x_3):
        x = torch.stack((x_1, x_2, x_3), dim=1)
        x = torch.unsqueeze(x, dim=1)
        out = self.conv3(x)
        out = torch.squeeze(out, dim=1)
        out = torch.squeeze(out, dim=1)
        out_feature = self.relu(out)
        out = self.fc1(out_feature)
        out_feature = self.relu(out)
        out = self.fc2(out_feature)
        out_feature = self.relu(out)
        out = self.fc3(out_feature)
        out_feature = self.relu(out)
        out = self.fc4(out_feature)
        out_feature = self.relu(out)
        out = self.fc5(out_feature)
        return out


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.2)
