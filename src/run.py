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

    print("Training gradually together with random uniform choice")
    print("Checking their accuracy on validation set")
    results_together_cleaned = {
        key: value[0][f"{key}_val_acc"] for key, value in results_together.items()
    }
    results_together_cleaned = sorted(
        results_together_cleaned.items(), key=lambda x: x[1]
    )
    pprint(results_together_cleaned)

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

    print("Checking accuracy on validation set of separate subnetworks")
    results_sep_cleaned = {
        key: value[0][f"{key}_val_acc"] for key, value in results_sep.items()
    }
    results_sep_cleaned = sorted(results_sep_cleaned.items(), key=lambda x: x[1])
    pprint(results_sep_cleaned)

    configurations_full = ["1111", "1010", "1001", "0110", "0101"]

    model = SuperNetMNIST(is_train_mult=False, flow_solo="1111")
    trainer = Trainer(
        gpus=AVAIL_GPUS, max_epochs=20, callbacks=[ModelCheckpoint("logs/overall")]
    )
    trainer.fit(model)

    results_full = {}
    for conf in configurations_full:
        model.sample_subnetwork(conf)
        results_full[conf] = trainer.test(model)

    print("Checking accuracy on validation set of full network")
    results_full_cleaned = {
        key: value[0][f"{key}_val_acc"] for key, value in results_full.items()
    }
    results_full_cleaned = sorted(results_full_cleaned.items(), key=lambda x: x[1])
    pprint(results_full_cleaned)
