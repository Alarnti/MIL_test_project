import os
from pprint import pprint

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from model import SuperNetMNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512 if AVAIL_GPUS else 64

SEED = 42
seed_everything(SEED, workers=True)


if __name__ == "__main__":
    configurations = ["1010", "1001", "0110", "0101"]

    model = SuperNetMNIST(is_train_mult=True)
    trainer = Trainer(
        gpus=AVAIL_GPUS, max_epochs=20, callbacks=[ModelCheckpoint("logs/together")]
    )
    trainer.fit(model)

    results_together = {}
    for conf in configurations:
        model.sample_subnetwork(conf)
        results_together[conf] = trainer.test(model)

    # print('Training gradually together with random uniform choice')
    # pprint(results_together)

    results_sep = {}
    for conf in configurations:
        model = SuperNetMNIST(is_train_mult=False, flow_solo=conf)
        trainer = Trainer(
            gpus=AVAIL_GPUS,
            max_epochs=20,
            callbacks=[ModelCheckpoint(f"logs/separately_{conf}")],
        )
        trainer.fit(model)
        results_sep[conf] = trainer.test(model)

    # print('Training separately')
    # pprint(results_sep)

    for conf in configurations:
        print("Configuration " + conf)
        print(
            results_together[conf][f"{conf}_val_acc"]
            - results_together[conf][f"{conf}_val_acc"]
        )
