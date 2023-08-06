import gc
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
import torch.functional as F
from trainers import setup_evaluator, setup_trainer
from utils import *

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

def get_model(config, domain, logger):
    checkpoint_root = os.path.join("./checkpoints", config.dataset.name)

    model = setup_model(config, return_feat_only=True)

    candidate_checkpoint = glob.glob(
        os.path.join(checkpoint_root, domain, "*.pt")
    )
    assert len(candidate_checkpoint) == 1
    checkpoint_path = candidate_checkpoint[0]

    to_save_eval = {"model": model}
    resume_from(to_save_eval, checkpoint_path, logger)

    if logger is not None:
        logger.info(f"checkpoint loaded from {checkpoint_path}")
    return model


def run(config: Any):
    # make a certain seed
    manual_seed(config.seed)
    logger = logging.getLogger()

    dataloader_train, dataloader_test = setup_data(
        config, is_test=True, few_shot_num=config.few_shot_num
    )

    num_classes = config.dataset.num_classes

    domains = config.dataset.domains
    target_domain = config.dataset.domain
    target_domain_index = -1

    for i, d in enumerate(domains):
        if d == target_domain:
            target_domain_index = i
            break
    assert target_domain_index != -1

    del domains[target_domain_index]

    # domains = ["v_task2", "v_task1"]
    # source_domains = [i for i in config.dataset.domains if i != config.dataset.domain]

    all_features_train = torch.zeros((len(domains), len(dataloader_train.dataset), config.model.hidden_dim))
    all_label_train = None

    print(f"target domain: {config.dataset.domain}")

    with torch.no_grad():

        # save labels & features
        for i, d in enumerate(domains):
            print(f"domain: {d}")
            
            model = get_model(config, d, logger).cuda().eval()

            print(f"extract features by {d}-trained model")
            features_list = []
            labels_list = []
            for data in tqdm(dataloader_train):
                # get the inputs
                inputs, labels = data

                inputs = inputs.cuda()
                # labels = labels.cuda()
                
                features_list.append(model(inputs).detach().cpu())
                if all_label_train is None:
                    labels_list.append(labels.detach().cpu())
            all_features_train[i, :, :] = torch.cat(features_list, dim=0)
            if all_label_train is None:
                all_label_train = torch.cat(labels_list, dim=0)
    # stop with torch.no_grad()

    del inputs, labels, data
    del dataloader_train
    del model
    del features_list, labels_list


    # compute H-score and validation accuracy
    print(f"\n\n*******start calc H-Score*******")
    
    def get_target_feature(alpha, all_features):
        # target_feature: sum of all features weighted by alpha
        target_feature = torch.zeros_like(all_features[0, :, :])
        for i in range(len(domains)):
            target_feature += alpha[i] * all_features[i, :, :]
        return target_feature

    def get_target_feature_train(alpha, all_features):
        target_feature = torch.zeros_like(all_features[0, :, :])
        for i in range(len(domains) - 1):
            target_feature += alpha[i] * all_features[i, :, :]
        i = len(domains) - 1
        target_feature += (1 - alpha.sum()) * all_features[i, :, :]
        return target_feature
    
    def get_score(features, labels):
        Covf = torch.cov(features.T)  # (hidden_dim, hidden_dim)
        label_choice = torch.unique(labels)
        g = torch.zeros_like(features)
        for z in label_choice:
            fl = features[labels == z, :]
            Ef_z = torch.mean(fl, dim=0) # (hidden_dim)
            g[labels == z] = Ef_z
        Covg = torch.cov(g.T)
        dif = torch.trace(Covg) / torch.trace(Covf)
        return dif
    
    alpha = torch.ones(len(domains) - 1) / len(domains)
    # alpha = torch.ones(len(domains)) / len(domains)
    # # insert 0 to alpha at target_domain_index
    # alpha = torch.cat([alpha[:target_domain_index], torch.zeros(1), alpha[target_domain_index:]], dim=0)

    alpha.requires_grad = True
    all_features_train.requires_grad = False
    all_label_train.requires_grad = False

    optimizer = optim.SGD([alpha], lr=0.005)
    # optimizer = optim.AdamW([alpha], lr=0.005)
    for epoch in range(1000):
        optimizer.zero_grad()

        target_feature = get_target_feature_train(alpha, all_features_train)
        # target_feature = get_target_feature(alpha, all_features_train)
        h_score = -get_score(target_feature, all_label_train)

        h_score.backward()
        optimizer.step()

        # print(f"epoch {epoch}: h_score {h_score.item()}")

        # alpha.requires_grad = False
        # alpha[alpha < 0] = 0
        # alpha = alpha / alpha.sum() if alpha.sum() > 1 else alpha
        # alpha.requires_grad = True
    
    alpha.requires_grad = False
    alpha = torch.cat([alpha, (1 - alpha.sum()).view(-1)], dim=0)
    print(f"final alpha: {alpha}")
    


    print(f"\n\n*******get G*******")
    with torch.no_grad():
        def normalize(features):
            return (features - features.mean(axis=0)) / features.std(axis=0)
        
        target_feature = get_target_feature(alpha, all_features_train)
        target_feature = normalize(target_feature)

        gamma_f = target_feature.T@target_feature / target_feature.shape[0]

        def get_conditional_exp(feature, label, num_classes):
            "calculate conditional expectation of fx"
            ce_f = torch.zeros((num_classes, feature.shape[1]))

            for i in range(num_classes):
                fx_i = feature[torch.where(label==i)] - feature.mean(0)
                ce_f[i] = fx_i.mean(axis=0)
            
            return ce_f

        # ce_f_s = torch.zeros((len(domains), num_classes, target_feature.shape[1]))
        ce_f_s_list = [
            get_conditional_exp(
                label=all_label_train,
                feature=all_features_train[i, :, :],
                num_classes=num_classes
            ) for i in range(len(domains))
        ]
        ce_f_s = torch.stack(ce_f_s_list, dim=0)

        # torch.permute = np.transpose ; torch.transpose = np.swapaxes; torch.mm = np.dot ; torch.inverse = np.linalg.inv; 
        # g = torch.inverse(gamma_f).mm((ce_f_s.permute((1,2,0)).mm(alpha)).T).T
        g = (
            torch.inverse(gamma_f) @ (ce_f_s.permute((1,2,0))@alpha).T
        ).T

    del all_features_train, ce_f_s, ce_f_s_list, gamma_f
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n\n*******start test on {config.dataset.domain}*******")
    with torch.no_grad():
        g_norm = normalize(g)
        f_norm = normalize(target_feature)
        score = f_norm @ g_norm.T
        acc_test = (torch.argmax(score, dim=1) == all_label_train).sum().item() / len(all_label_train)
        print("Train accuracy: ", acc_test)

        del g_norm, f_norm, score, target_feature, all_label_train

        # test_features = torch.zeros(len(dataloader_test.dataset), config.model.hidden_dim).cuda()
        features_list = []
        labels_list = []
        acc_test = 0

        for data in tqdm(dataloader_test):
            inputs, labels = data
            inputs = inputs.cuda()

            labels_list.append(labels.detach().cpu())
            features_list.append(torch.zeros(inputs.shape[0], config.model.hidden_dim).cuda())

            for i, d in enumerate(domains):
            
                model = get_model(config, d, None).cuda().eval()

                features = model(inputs).detach()
                features = normalize(features)
                
                features_list[-1] += features * alpha[i]

        test_features = torch.cat(features_list, dim=0)
        test_label = torch.cat(labels_list, dim=0)

        del model, features_list, labels_list, features, labels, data, inputs
        g_norm = normalize(g.cuda())
        f_norm = normalize(test_features.cuda())
        del test_features,
        score = f_norm @ g_norm.T
        print("Correct num: ", (torch.argmax(score, dim=1) == test_label.cuda()).sum().cpu().item())
        print("Incorrect num: ", (torch.argmax(score, dim=1) != test_label.cuda()).sum().cpu().item())
        acc_test = (torch.argmax(score, dim=1) == test_label.cuda()).sum().cpu().item() / len(dataloader_test.dataset)
        print(f"Test accuracy: {acc_test} ; test sample: {len(dataloader_test.dataset)}")
    print(f"*******done {config.dataset.domain} of {domains}*******")
    print("#########################################################\n\n\n")


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
