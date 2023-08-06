import glob
import os
from pprint import pformat
from typing import Any

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

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf


def run(config: Any):
    # make a certain seed
    manual_seed(config.seed)
    logger = logging.getLogger()

    dataloader_train, dataloader_test = setup_data(
        config, is_test=True, few_shot_num=config.few_shot_num
    )

    num_classes = config.dataset.num_classes

    domains = config.dataset.domains
    # domains = config.dataset.domains[:4]
    # domains = ["v_task2", "v_task1"]

    # source_domains = [i for i in config.dataset.domains if i != config.dataset.domain]

    sigma = torch.zeros((len(domains), config.model.hidden_dim)).cuda()
    g = torch.zeros((num_classes, config.model.hidden_dim, len(domains))).cuda()

    checkpoint_root = os.path.join("./checkpoints", config.dataset.name)

    print(f"target domain: {config.dataset.domain}")

    with torch.no_grad():
        running_probs_train = torch.zeros((len(dataloader_train.dataset), num_classes)).cuda()
        running_probs_test = torch.zeros((len(dataloader_test.dataset), num_classes)).cuda()
        for i, d in enumerate(domains):
            if d == config.dataset.domain:
                continue

            print(f"domain: {d}")
            
            model = setup_model(config, return_feat_only=True)

            candidate_checkpoint = glob.glob(
                os.path.join(checkpoint_root, d, "*.pt")
            )
            assert len(candidate_checkpoint) == 1
            checkpoint_path = candidate_checkpoint[0]

            to_save_eval = {"model": model}
            resume_from(to_save_eval, checkpoint_path, logger)

            print(f"checkpoint loaded from {checkpoint_path}")

            model.eval()
            model.cuda()

            print(f"extract features by {d}-trained model")
            for data in tqdm(dataloader_train):
                # get the inputs
                inputs, labels = data

                inputs = inputs.cuda()
                labels = labels.cuda()

                # sigma = torch.zeros((len(domains), config.model.hidden_dim))
                # g = torch.zeros((num_classes, config.model.hidden_dim, len(domains)))
                sigma[i, :], g[:, :, i] = compute_max_corr(
                    model, inputs, labels, num_classes
                )
                res = weighted_network_output( # only one batch! so this is correct
                    model, sigma[i, :], g[:, :, i], inputs
                )
                
                running_probs_train += res
                
                tqdm.write(f"acc: {(torch.argmax(res, dim=1) == labels.long()).float().mean().item()}")

            r = 0
            for batch_index, data in enumerate(tqdm(dataloader_test)):
                index_start = batch_index * dataloader_test.batch_size
                index_end = min(index_start + dataloader_test.batch_size, len(dataloader_test.dataset))

                inputs, labels = data

                inputs = inputs.cuda()
                labels = labels.cuda()

                res = weighted_network_output(
                    model, sigma[i, :], g[:, :, i], inputs
                )
                running_probs_test[index_start:index_end] += res

                # tqdm.write(f"acc: {(torch.argmax(res, dim=1) == labels.long()).sum().item()/labels.size(0)} label: {labels.long()[0]}")

                r += (torch.argmax(res, dim=1) == labels.long()).sum().item()
            print(f"acc: {r/len(dataloader_test.dataset)}")

        # all_feats_train = torch.zeros(
        #     (len(dataloader_train.dataset), config.model.hidden_dim * len(domains))
        # )
        # all_feats_test = torch.zeros((len(testloader.dataset), config.model.hidden_dim * len(domains)))
        # for i in range(len(domains)):
        #     for data in dataloader_train:
        #         # get the inputs
        #         inputs, labels_train = data
        #         all_feats_train[:, i * config.model.hidden_dim : (i + 1) * config.model.hidden_dim] = all_nets[i][0](inputs)
        #     for data in testloader:
        #         inputs, labels_test = data
        #         all_feats_test[:, i * config.model.hidden_dim : (i + 1) * config.model.hidden_dim] = all_nets[i][0](inputs)

        print(f"\n\n*******start test on {config.dataset.domain}*******")


        running_probs_train = running_probs_train.cpu()
        _, predicted_test = torch.max(running_probs_train, 1)
        acc_test = 0
        for batch_index, data in enumerate(tqdm(dataloader_train)):
            index_start = batch_index * dataloader_train.batch_size
            index_end = min(index_start + dataloader_train.batch_size, len(dataloader_train.dataset))

            inputs, labels = data

            acc_test += (
                (predicted_test[index_start:index_end] == labels.long()).sum().item()
            )  # /labels.size(0)
        
        print("Train accuracy: ", acc_test / len(dataloader_train.dataset))

        running_probs_test = running_probs_test.cpu()
        _, predicted_test = torch.max(running_probs_test, 1)
        acc_test = 0
        for batch_index, data in enumerate(tqdm(dataloader_test)):
            index_start = batch_index * dataloader_test.batch_size
            index_end = min(index_start + dataloader_test.batch_size, len(dataloader_test.dataset))

            inputs, labels = data

            acc_test += (
                (predicted_test[index_start:index_end] == labels.long()).sum().item()
            )  # /labels.size(0)
        
        print("Test accuracy: ", acc_test / len(dataloader_test.dataset))




def compute_max_corr(net, data, labels, num_classes):
    """Computes maximal correlation and also returns the associated g(y)"""
    outputs = net(data)
    outputs -= outputs.mean(dim=0)
    outputs /= get_std_devs(net, data)

    # outputs = outputs.cpu()

    g_y = torch.zeros((num_classes, outputs.shape[1])).cuda()
    for idx, row in enumerate(outputs.split(1)):
        g_y[labels[idx]] += row.detach().reshape(-1)
    g_y /= labels.shape[0]
    sigma = torch.zeros(outputs.shape[1]).cuda()
    for idx, row in enumerate(outputs.split(1)):
        sigma += row.detach().reshape(-1) * g_y[labels[idx], :]
    sigma /= labels.shape[0]
    # make sure signs are positive
    g_y *= sigma.sign()
    sigma *= sigma.sign()
    return sigma, g_y


def weighted_network_output(net, sigma, g, inputs):
    """Output weighted sum(sigma*f*g) for both values of g"""
    outputs = net(inputs)
    outputs -= outputs.mean(dim=0)

    # outputs = outputs.cpu()

    outputs *= sigma.reshape(1, -1)
    outputs = torch.mm(outputs, g.permute(1, 0))
    return outputs


def get_means(net, inputs):
    outputs = net(inputs)
    return outputs.mean(dim=0)


def get_std_devs(net, inputs):
    outputs = net(inputs)
    outputs -= outputs.mean(dim=0)
    stds = torch.sqrt(torch.diag(torch.mm(outputs.permute(1, 0), outputs)))
    stds[stds == 0] = 1

    return stds


# main entrypoint
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # with idist.Parallel() as p:
    #     # with idist.Parallel("gloo") as p:
    #     p.run(run, config=cfg)

    

    run(cfg)


if __name__ == "__main__":
    # CUBLAS_WORKSPACE_CONFIG=:4096:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    main()
