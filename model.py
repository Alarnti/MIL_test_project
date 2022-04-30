from __future__ import annotations

import numpy as np
import torch
from pytorch_lightning import LightningModule, seed_everything
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, F1Score
from torchvision import transforms
from torchvision.datasets import MNIST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 512 if torch.cuda.is_available() else 64

SEED = 42
seed_everything(SEED, workers=True)


class SuperNetMNIST(LightningModule):
    def __init__(
        self,
        is_train_mult: str = True,
        flow_solo: str = "1111",
        data_dir: str = ".",
        batch_size: int = 512,
        ch_size: int = 2,
        learning_rate: float = 5e-4,
    ):
        super().__init__()

        self.data_dir = data_dir

        self.ch_size = ch_size
        self.batch_size = batch_size

        self.num_classes = 10

        self.learning_rate = learning_rate

        self.conv1_3x3 = nn.Conv2d(1, self.ch_size, 3, padding=1)
        self.conv1_5x5 = nn.Conv2d(1, self.ch_size, 5, padding=2)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv2_3x3 = nn.Conv2d(ch_size, self.ch_size * 2, 3, padding=1)
        self.conv2_5x5 = nn.Conv2d(ch_size, self.ch_size * 2, 5, padding=2)

        self.fc1 = nn.Linear(2 * self.ch_size * 7 * 7, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

        self.train_multiple = is_train_mult
        self.all_flows = ["1010", "1001", "0110", "0101"]
        self.current_flow = flow_solo

        # Hardcode some dataset specific attributes

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # TODO ???????????????????????
            ]
        )

        self.accuracy = Accuracy()
        self.F1_score = F1Score(num_classes=self.num_classes, average="macro")

    def forward(self, x: Tensor) -> Tensor:
        x1 = torch.zeros(x.size(0), self.ch_size, x.size(2), x.size(3)).to(DEVICE)
        if self.current_flow[0] == "1":
            x1 += torch.relu(self.conv1_3x3(x))
        if self.current_flow[1] == "1":
            x1 += torch.relu(self.conv1_5x5(x))

        x1 = self.maxpool(x1)

        x2 = torch.zeros(x1.size(0), self.ch_size * 2, x1.size(2), x1.size(3)).to(
            DEVICE
        )
        if self.current_flow[2] == "1":
            x2 += F.relu(self.conv2_3x3(x1))
        if self.current_flow[3] == "1":
            x2 += F.relu(self.conv2_5x5(x1))

        x2 = self.maxpool(x2)

        x2_fc = x2.view(-1, x2.size(1) * x2.size(2) * x2.size(3))

        x2_fc = F.relu(self.fc1(x2_fc))

        return F.log_softmax(self.fc2(x2_fc), dim=1)

    def training_step(self, batch: list[Tensor, Tensor], batch_idx: Tensor) -> Tensor:
        if self.train_multiple:
            self.current_flow = np.random.choice(self.all_flows, 1)[0]
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)

        self.log(self.current_flow + "_train_loss", loss)

        return loss

    def validation_step(self, batch: list[Tensor, Tensor], batch_idx: Tensor) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.F1_score(preds, y)

        self.log(self.current_flow + "_val_loss", loss)
        self.log(self.current_flow + "_f1_score", self.F1_score)
        self.log(self.current_flow + "_val_acc", self.accuracy)
        return loss

    def test_step(self, batch: list[Tensor, Tensor], batch_idx: Tensor) -> Tensor:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def sample_subnetwork(
        self, sampled_net: str = "1010", path_pretrained: str = None
    ) -> SuperNetMNIST:
        self.current_flow = sampled_net
        if path_pretrained is not None:
            return self.load_from_checkpoint(path_pretrained)
        else:
            return self

    #     def on_after_backward(self):
    #         global_step = self.global_step
    #         for name, param in self.state_dict().items():
    #             self.logger.experiment.add_histogram(name, param, global_step)
    #             if param.requires_grad:
    #                 self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, shuffle=True, num_workers=12, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, num_workers=12, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, num_workers=12, batch_size=self.batch_size)
