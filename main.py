import os
from pprint import pformat
from typing import Any

import ignite.distributed as idist
from ignite.engine import create_supervised_evaluator
import yaml
from data import setup_data
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed
from models import setup_model
from torch import nn, optim
from trainers import setup_evaluator, setup_trainer
from utils import *

import hydra
from omegaconf import DictConfig, OmegaConf


def run(local_rank: int, config: Any):
    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    # if rank == 0:
        # from clearml import Task
        # task = Task.init(project_name='multi-source', task_name='Office-Home Source Model Training')

    # create output folder
    config.output_dir = setup_output_dir(config, rank)

    # donwload datasets and create dataloaders
    dataloader_train, dataloader_eval = setup_data(config)

    # model, optimizer, loss function, device
    device = idist.device()
    model = idist.auto_model(setup_model(config))
    optimizer = idist.auto_optim(optim.AdamW(model.parameters(), lr=config.lr))
    loss_fn = nn.CrossEntropyLoss().to(device=device)

    # trainer and evaluator
    trainer = setup_trainer(
        config, model, optimizer, loss_fn, device, dataloader_train.sampler
    )
    evaluator = setup_evaluator(config, model, device)

    # from torch.optim.lr_scheduler import ExponentialLR
    # from torch.optim import lr_scheduler
    # from ignite.contrib.handlers import LRScheduler, create_lr_scheduler_with_warmup
    # # step_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
    # step_scheduler=lr_scheduler.CyclicLR(
    #     optimizer,base_lr=0.00002,max_lr=0.0002,step_size_up=30,step_size_down=30, cycle_momentum=False
    # )
    # # scheduler = LRScheduler(step_scheduler)
    # scheduler = create_lr_scheduler_with_warmup(step_scheduler,
    #                                             warmup_start_value=0.0002,
    #                                             warmup_end_value=0.0002,
    #                                             warmup_duration=30*20)

    # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    

    # from ignite.handlers import FastaiLRFinder
    # lr_finder = FastaiLRFinder()
    # to_save = {"model": model, "optimizer": optimizer}

    # with lr_finder.attach(trainer, to_save=to_save, start_lr=1e-6) as trainer_with_lr_finder:
    #     trainer_with_lr_finder.run(dataloader_train)

    # # Get lr_finder results
    # print(lr_finder.get_results())

    # # Plot lr_finder results (requires matplotlib)
    # # lr_finder.plot()

    # ax = lr_finder.plot(skip_end=0, skip_start=0)
    # ax.figure.savefig("output.jpg")

    # # get lr_finder suggestion for lr
    # print(lr_finder.lr_suggestion())
    # exit()

    # attach metrics to evaluator
    accuracy = Accuracy(device=device)
    metrics = {
        "eval_accuracy": accuracy,
        "eval_loss": Loss(loss_fn, device=device),
        # "eval_error": (1.0 - accuracy) * 100,
    }
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    train_evaluator = create_supervised_evaluator(model, metrics = {
        "train_accuracy": Accuracy(device=device),
        "train_loss": Loss(loss_fn, device=device),
    }, device=device)

    # setup engines logger with python logging
    # print training configurations
    logger = setup_logging("ignite", config)
    logger.info("Configuration: \n%s", pformat(vars(config)))
    (config.output_dir / "config-lock.yaml").write_text(yaml.dump(OmegaConf.to_yaml(config)))

    trainer.logger = setup_logging("trainer", config)
    evaluator.logger = setup_logging("evaluator", config)
    train_evaluator.logger = setup_logging("train_evaluator", config)

    # setup ignite handlers
    to_save_train = {"model": model, "optimizer": optimizer, "trainer": trainer}
    to_save_eval = {"model": model}
    ckpt_handler_train, ckpt_handler_eval = setup_handlers(
        trainer, evaluator, config, to_save_train, to_save_eval
    )
    # experiment tracking
    if rank == 0:
        exp_logger = setup_exp_logging(config, trainer, optimizer, {
            "eval_evaluator": evaluator, 
            "train_evaluator": train_evaluator
        })

    # print metrics to the stderr
    # with `add_event_handler` API
    # for training stats
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_iters),
        log_metrics,
        tag="train",
    )

    # run evaluation at every training epoch end
    # with shortcut `on` decorator API and
    # print metrics to the stderr
    # again with `add_event_handler` API
    # for evaluation stats
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def _():
        train_evaluator.run(dataloader_train)
        log_metrics(train_evaluator, "train_evaluator")
        evaluator.run(dataloader_eval)
        log_metrics(evaluator, "eval")

    # let's try run evaluation first as a sanity check
    @trainer.on(Events.STARTED)
    def _():
        evaluator.run(dataloader_eval)
        log_metrics(evaluator, "init_eval")

    # setup if done. let's run the training
    trainer.run(
        dataloader_train,
        max_epochs=config.max_epochs,
    )
    # close logger
    if rank == 0:
        exp_logger.close()

    # show last checkpoint names
    logger.info(
        "Last training checkpoint name - %s",
        ckpt_handler_train.last_checkpoint,
    )

    logger.info(
        "Last evaluation checkpoint name - %s",
        ckpt_handler_eval.last_checkpoint,
    )


# main entrypoint
@hydra.main(version_base=None, config_path="conf", config_name="config")    
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    with idist.Parallel() as p:
    # with idist.Parallel("gloo") as p:
        p.run(run, config=cfg)


if __name__ == "__main__":

    # CUBLAS_WORKSPACE_CONFIG=:4096:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    main()
