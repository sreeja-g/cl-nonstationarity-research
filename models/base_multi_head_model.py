import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMultiHeadModel(nn.Module):
    """
    Base CNN model with task-specific heads.
    Designed for task-incremental learning (task-id available).
    """

    def __init__(
        self,
        num_classes_per_task: dict[int, int],
        input_channels: int = 3,
        input_size: int = 32,
    ):
        super().__init__()

        self.num_classes_per_task = num_classes_per_task

        # -----------------------
        # Feature extractor
        # -----------------------
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # -----------------------
        # Dynamically compute feature size
        # -----------------------
        feature_dim = self._get_feature_dim(
            input_channels=input_channels,
            input_size=input_size,
        )

        # -----------------------
        # Shared representation
        # -----------------------
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 512)

        # -----------------------
        # Task-specific heads
        # -----------------------
        self.task_heads = nn.ModuleDict({
            str(task_id): nn.Linear(512, num_classes)
            for task_id, num_classes in num_classes_per_task.items()
        })

    # -----------------------
    # Infer CNN output size
    # -----------------------
    def _get_feature_dim(
        self,
        input_channels,
        input_size,
    ):
        with torch.no_grad():
            x = torch.zeros(
                1,
                input_channels,
                input_size,
                input_size,
            )

            x = self.features(x)

            return x.view(1, -1).size(1)

    # -----------------------
    # Feature extractor
    # -----------------------
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    # -----------------------
    # Forward
    # -----------------------
    def forward(
        self,
        x: torch.Tensor,
        task_id,
    ) -> torch.Tensor:

        x = self.extract_features(x)

        return self.task_heads[str(task_id)](x)

    # -----------------------
    # Utility
    # -----------------------
    def get_head(self, task_id):
        return self.task_heads[str(task_id)]