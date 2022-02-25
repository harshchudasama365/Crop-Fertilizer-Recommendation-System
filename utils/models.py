import torch 
import torch.nn as nn
import torch.nn.functional as F


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # [N, 32, (256-3+2)+1, (256-3+2)+1] = [N, 32, 256, 256]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # [N, 32, (256-3+2)+1, (256-3+2)+1] = [N, 32, 256, 256]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),                                                      # [N, 32, 256/2, 256/2] = [N, 32, 128, 128]
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # [N, 64, (128-3+2)+1, (128-3+2)+1] = [N, 64, 128, 128]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # [N, 64, (128-3+2)+1, (128-3+2)+1] = [N, 64, 128, 128]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                                                      # [N, 64, 128/2, 128/2] = [N, 64, 64, 64]
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),# [N, 128, (64-3+2)+1, (64-3+2)+1] = [N, 128, 64, 64]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),# [N, 128, (64-3+2)+1, (64-3+2)+1] = [N, 128, 64, 64]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                                                       # [N, 128, 64/2, 64/2] = [N, 128, 32, 32]
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),# [N, 256, (32-3+2)+1, (32-3+2)+1] = [N, 256, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),# [N, 256, (32-3+2)+1, (32-3+2)+1] = [N, 256, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),                                                       #[N, 256, 32/2, 32/2] = [N, 256, 16, 16]
            # conv5
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),# [N, 512, (16-3+2)+1, (16-3+2)+1] = [N, 512, 16, 16]
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)                                                        #[N, 512, 16/2, 16/2] = [N, 512, 8, 8]
        )

        self.dense_layers = nn.Sequential(
            nn.Flatten(),                # [N, 512*8*8] = [N, 32767]
            nn.Dropout(0.4),
            nn.Linear(32768, 1024),      # [N, 1024]
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 38)          # [N, 38]
        )

    def forward(self, X):
        # Convolution layers
        out = self.conv_layers(X)
        
        # Fully connected layers
        out = self.dense_layers(out)

        return out