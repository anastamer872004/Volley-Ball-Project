import torch
from torch import nn


class Baseline8(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size1,
        hidden_size2,
        num_layers,
        num_classes,
    ):
        super().__init__()

        # ============================================================
        # LSTM 1: Player-level temporal modeling
        # ============================================================
        # Each player has a sequence of 9 frames.
        #
        # Input:
        #     (B, T, F)
        #
        # Output:
        #     (B, T, H1)
        #
        # Example:
        #     (768, 9, 2048) -> (768, 9, 2048)
        #
        # because hidden_size1 = 2048 in the current trainer.
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=num_layers,
            batch_first=True,
        )

        # ============================================================
        # LSTM 2: Frame/scene-level temporal modeling
        # ============================================================
        # Team 1 representation = hidden_size1
        # Team 2 representation = hidden_size1
        #
        # Concatenation:
        #     hidden_size1 + hidden_size1
        #     = hidden_size1 * 2
        #
        # Therefore LSTM2 input_size = hidden_size1 * 2.
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1 * 2,
            hidden_size=hidden_size2,
            num_layers=num_layers,
            batch_first=True,
        )

        # ============================================================
        # Final classifier
        # ============================================================
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
            T = number of frames
            P = number of players
            F = feature dimension

        Example:
            (64, 9, 12, 2048)

        Output:
            (B, num_classes)

        Example:
            (64, 8)
        """

        # ------------------------------------------------------------
        # Initial dimensions
        # ------------------------------------------------------------
        batch_size, sequence_length, num_players, num_features = x.shape

        # ============================================================
        # 1. Prepare data for LSTM1
        # ============================================================
        # Original:
        #     (B, T, P, F)
        #
        # We need:
        #     (B, P, T, F)
        #
        # because each player should have its own 9-frame sequence.
        x = x.permute(0, 2, 1, 3).contiguous()

        # Merge batch and player dimensions:
        #
        #     (B, P, T, F)
        #          ↓
        #     (B*P, T, F)
        #
        # Example:
        #     (64, 12, 9, 2048)
        #          ↓
        #     (768, 9, 2048)
        x = x.view(
            batch_size * num_players,
            sequence_length,
            num_features,
        )

        # ============================================================
        # 2. LSTM1: process each player's temporal sequence
        # ============================================================
        #
        # Input:
        #     (B*P, T, F)
        #
        # Output:
        #     (B*P, T, H1)
        #
        # Example:
        #     (768, 9, 2048)
        #          ↓
        #     (768, 9, 2048)
        x, _ = self.lstm1(x)

        # ============================================================
        # 3. Restore player dimension
        # ============================================================
        #
        # (B*P, T, H1)
        #       ↓
        # (B, P, T, H1)
        #
        # Example:
        # (768, 9, 2048)
        #       ↓
        # (64, 12, 9, 2048)
        x = x.view(
            batch_size,
            num_players,
            sequence_length,
            -1,
        )

        # ============================================================
        # 4. Split players into two teams
        # ============================================================
        #
        # Assumption:
        #     players 0-5  -> Team 1
        #     players 6-11 -> Team 2
        #
        # Team 1:
        #     (B, 6, T, H1)
        #
        # Team 2:
        #     (B, 6, T, H1)

        team1 = x[:, :6, :, :]
        team2 = x[:, 6:, :, :]

        # ============================================================
        # 5. Pool players within each team
        # ============================================================
        #
        # We want to pool the 6 players separately FOR EACH FRAME.
        #
        # Team 1:
        #     (B, 6, T, H1)
        #          ↓ max over players (dim=1)
        #     (B, T, H1)
        #
        # Same for Team 2.

        team1 = torch.max(team1, dim=1).values
        team2 = torch.max(team2, dim=1).values

        # Now:
        #
        # team1 = (B, T, H1)
        # team2 = (B, T, H1)

        # ============================================================
        # 6. Concatenate the two team representations
        # ============================================================
        #
        # For every frame:
        #
        # Team 1: H1
        # Team 2: H1
        #
        # Combined:
        #     H1 + H1 = 2*H1
        #
        # Example:
        #     (64, 9, 2048)
        #     +
        #     (64, 9, 2048)
        #
        #     -> (64, 9, 4096)

        x = torch.cat((team1, team2), dim=2)

        # ============================================================
        # 7. LSTM2: frame-level / scene-level temporal modeling
        # ============================================================
        #
        # Input:
        #     (B, T, 2*H1)
        #
        # Output:
        #     (B, T, H2)
        #
        # Example:
        #     (64, 9, 4096)
        #          ↓
        #     (64, 9, 2048)

        x, _ = self.lstm2(x)

        # ============================================================
        # 8. Take final time step
        # ============================================================
        #
        # (B, T, H2)
        #      ↓
        # (B, H2)
        #
        # Example:
        # (64, 9, 2048)
        #      ↓
        # (64, 2048)

        x = x[:, -1, :]

        # ============================================================
        # 9. Final classification
        # ============================================================
        #
        # (B, H2)
        #      ↓
        # (B, 512)
        #      ↓
        # (B, 8)

        return self.fc(x)