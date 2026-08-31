import torch
from torch import nn


class Baseline7(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size1,
        hidden_size2,
        num_layers,
        num_classes,
    ):
        super().__init__()

        # LSTM 1:
        # Processes the temporal sequence of EACH PLAYER independently.
        # Input: 2048 features per frame
        # Output: 1024 hidden features per frame
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=num_layers,
            batch_first=True,
        )

        # LSTM 2:
        # Processes the sequence of FRAME-LEVEL representations.
        # Input: 1024 features per frame
        # Output: 1024 hidden features per frame
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=num_layers,
            batch_first=True,
        )

        # Final group-activity classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Input:
            x: (B, T, P, F)

            B = batch size
            T = sequence length / number of frames
            P = number of players
            F = number of features per player

        Example:
            (64, 9, 12, 2048)
        """

        batch_size, sequence_length, num_players, num_features = x.shape

        # ============================================================
        # 1. LSTM1: Temporal modeling at PLAYER level
        # ============================================================

        # Original:
        # (B, T, P, F)
        #
        # We need:
        # (B, P, T, F)
        #
        # So each player gets its own sequence of 9 frames.
        x = x.permute(0, 2, 1, 3).contiguous()

        # Merge batch and player dimensions:
        #
        # (B, P, T, F)
        #       ↓
        # (B*P, T, F)
        #
        # Example:
        # (64, 12, 9, 2048)
        #       ↓
        # (768, 9, 2048)
        x = x.view(
            batch_size * num_players,
            sequence_length,
            num_features,
        )

        # LSTM1 processes each player's 9-frame sequence
        #
        # (B*P, T, F)
        #       ↓
        # (B*P, T, H1)
        #
        # Example:
        # (768, 9, 2048)
        #       ↓
        # (768, 9, 1024)
        x, _ = self.lstm1(x)

        # ============================================================
        # 2. Restore player dimension
        # ============================================================

        # (B*P, T, H1)
        #       ↓
        # (B, P, T, H1)
        #
        # Example:
        # (768, 9, 1024)
        #       ↓
        # (64, 12, 9, 1024)
        x = x.view(
            batch_size,
            num_players,
            sequence_length,
            -1,
        )

        # ============================================================
        # 3. Max Pool players FOR EACH FRAME
        # ============================================================

        # We want:
        #
        # For Frame 1:
        #   Player 1 ─┐
        #   Player 2  │
        #   ...       ├── Max → Frame 1 representation
        #   Player 12 ┘
        #
        # Repeat for all 9 frames.
        #
        # Current:
        # (B, P, T, H1)
        #
        # Rearrange to:
        # (B, T, H1, P)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Merge batch and frame dimensions:
        #
        # (B, T, H1, P)
        #       ↓
        # (B*T, H1, P)
        #
        # Example:
        # (64, 9, 1024, 12)
        #       ↓
        # (576, 1024, 12)
        x = x.view(
            batch_size * sequence_length,
            -1,
            num_players,
        )

        # Max pooling over players
        #
        # (B*T, H1, P)
        #       ↓
        # (B*T, H1, 1)
        x = torch.max(x, dim=2, keepdim=True).values

        # Remove the last dimension
        #
        # (B*T, H1, 1)
        #       ↓
        # (B*T, H1)
        x = x.squeeze(-1)

        # Restore the temporal sequence
        #
        # (B*T, H1)
        #       ↓
        # (B, T, H1)
        #
        # Example:
        # (576, 1024)
        #       ↓
        # (64, 9, 1024)
        x = x.view(
            batch_size,
            sequence_length,
            -1,
        )

        # ============================================================
        # 4. LSTM2: Temporal modeling at FRAME level
        # ============================================================

        # Input:
        # (B, T, H1)
        #
        # Output:
        # (B, T, H2)
        #
        # Example:
        # (64, 9, 1024)
        #       ↓
        # (64, 9, 1024)
        x, _ = self.lstm2(x)

        # ============================================================
        # 5. Take the final time step
        # ============================================================

        # (B, T, H2)
        #       ↓
        # (B, H2)
        #
        # The final LSTM output summarizes the sequence.
        x = x[:, -1, :]

        # ============================================================
        # 6. Group activity classification
        # ============================================================

        # (B, 1024)
        #       ↓
        # (B, 512)
        #       ↓
        # (B, 8)
        return self.fc(x)