import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import matplotlib.pyplot as plt
from ResNet import CIFAR3_DataModule,CIFAR3_ResNet
from lab_tools import CIFAR10

dataset = CIFAR10('CIFAR10/')

def train(name, datamodule, model):

    tb_logger = TensorBoardLogger(".", name=name)
    callbacks = [TQDMProgressBar(refresh_rate=10)]

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=".",
    )

    trainer.fit(model, datamodule)


cifar3_datamodule = CIFAR3_DataModule(
    dataset=dataset,
    batch_size=100
)

resnet_model = CIFAR3_ResNet(lr=0.0001)

train(
    name="ResNet",
    datamodule=cifar3_datamodule,
    model=resnet_model,
)

tester = pl.Trainer(logger=False, enable_checkpointing=False)
tester.test(resnet_model, cifar3_datamodule)