import torch
import torch.nn as nn
import torch.nn.functional as F

from .partialconv2d import PartialConv2d


class VoiceFilter(nn.Module):
    def __init__(self, hp):
        super(VoiceFilter, self).__init__()
        self.hp = hp
        assert hp.audio.n_fft // 2 + 1 == hp.audio.num_freq == hp.model.fc2_dim, \
            "stft-related dimension mismatch"

        self.conv = nn.Sequential(
            # cnn1
            PartialConv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1), padding=(0, 3)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn2
            PartialConv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1), padding=(3, 0)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn3
            PartialConv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn4
            PartialConv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1), padding=(4, 2)), # (9, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn5
            PartialConv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1), padding=(8, 2)), # (17, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn6
            PartialConv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1), padding=(16, 2)), # (33, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn7
            PartialConv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1), padding=(32, 2)), # (65, 5)
            nn.BatchNorm2d(64), nn.ReLU(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            8*hp.audio.num_freq + hp.embedder.emb_dim,
            hp.model.lstm_dim,
            batch_first=True,
            bidirectional=True)

        self.fc1 = nn.Linear(2*hp.model.lstm_dim, hp.model.fc1_dim)
        self.fc2 = nn.Linear(hp.model.fc1_dim, hp.model.fc2_dim)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2) # [B, T, 8*num_freq + emb_dim]

        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x
