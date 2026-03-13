import torch.nn as nn

class ClassificationModel3D(nn.Module):
    def __init__(
            self,
            p
    ):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3, 1, 1)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_1_bn = nn.BatchNorm3d(8)

        self.Conv_2 = nn.Conv3d(8, 16, 3, 1, 1)
        self.Conv_2_mp = nn.MaxPool3d(2)
        self.Conv_2_bn = nn.BatchNorm3d(16)

        self.Conv_3 = nn.Conv3d(16, 24, 3, 1, 1)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_3_bn = nn.BatchNorm3d(24)

        self.Conv_4 = nn.Conv3d(24, 32, 3, 1, 1)
        self.Conv_4_mp = nn.MaxPool3d(2)
        self.Conv_4_bn = nn.BatchNorm3d(32)

        self.dense_1 = nn.LazyLinear(64)
        self.dense_2 = nn.Linear(64, 32)
        self.dense_3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1_mp(self.Conv_1(x))))
        x = self.relu(self.Conv_2_bn(self.Conv_2_mp(self.Conv_2(x))))
        x = self.relu(self.Conv_3_bn(self.Conv_3_mp(self.Conv_3(x))))
        x = self.relu(self.Conv_4_bn(self.Conv_4_mp(self.Conv_4(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.relu(self.dense_1(x))

        x = self.dropout(x)
        x = self.relu(self.dense_2(x))

        x = self.dense_3(x)

        return x

class ClassificationModel3D_inf(nn.Module):
    def __init__(
            self
    ):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3, 1, 1)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_1_bn = nn.BatchNorm3d(8)

        self.Conv_2 = nn.Conv3d(8, 16, 3, 1, 1)
        self.Conv_2_mp = nn.MaxPool3d(2)
        self.Conv_2_bn = nn.BatchNorm3d(16)

        self.Conv_3 = nn.Conv3d(16, 24, 3, 1, 1)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_3_bn = nn.BatchNorm3d(24)

        self.Conv_4 = nn.Conv3d(24, 32, 3, 1, 1)
        self.Conv_4_mp = nn.MaxPool3d(2)
        self.Conv_4_bn = nn.BatchNorm3d(32)

        self.dense_1 = nn.LazyLinear(64)
        self.dense_2 = nn.Linear(64, 32)
        self.dense_3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1_mp(self.Conv_1(x))))
        x = self.relu(self.Conv_2_bn(self.Conv_2_mp(self.Conv_2(x))))
        x = self.relu(self.Conv_3_bn(self.Conv_3_mp(self.Conv_3(x))))
        x = self.relu(self.Conv_4_bn(self.Conv_4_mp(self.Conv_4(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.relu(self.dense_1(x))

        x = self.dropout(x)
        x = self.relu(self.dense_2(x))

        x = self.dense_3(x)

        return x