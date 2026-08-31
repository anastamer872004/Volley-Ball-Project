import torch
from torch import nn


class Baseline4(nn.Module):
    def __init__(
        self,
        input_size=2048,
        hidden_size=1024,
        num_layers=1,
        num_classes=8,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, P, F)
        # B = batch ( Clip )
        # T = frames
        # P = players
        # F = features

        # Aggregate players for each frame
        x = torch.max(x, dim=2).values
        # (B, T, F)

        # Learn temporal relationships
        x, _ = self.lstm(x)
        # (B, T, H)

        # Use the representation from the final time step
        x = x[:, -1, :]
        # (B, H)

        # Group-activity classification
        x = self.fc(x)
        # (B, num_classes)

        return x



# import torch
# from torch import nn


# class Baseline4(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes),
#         )
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         # x shape: (batch_size, sequence_length, num_players, num_features)
#         x = torch.max(x, dim=2)[0]  # (batch_size, sequence_length, num_features)
#         x, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size)
#         x = x[:, -1, :]  # (batch_size, hidden_size)
#         return self.fc(x)  # (batch_size, num_classes)
