import glob
import os
import re
import shutil
import omegaconf

checkpoint_dir = "./checkpoints/"
logdir_dir = "./logs/"
dataset_conf_dir = "./conf/dataset/"
dataset = "visda2017"


cfg = omegaconf.OmegaConf.load(os.path.join(dataset_conf_dir, dataset + ".yaml"))

domains = cfg.domains

# There are many running log folders named like `20230729-215200-visda2017-t_task0`( {date-time-datasest-domain}-) in `logdir_dir`
# find the latest folders for each domain and get the latest checkpoint file name
# e.g. `20230729-215200-visda2017-t_task0/checkpoints/best_model_2_eval_accuracy=0.9987.pt`
for domain in domains:
    candidate_folders = glob.glob(
        "*-" + dataset + "-" + domain,
        root_dir=logdir_dir,
    )
    # latest_folder = max(candidate_folders, key=os.path.getctime)
    latest_folder = max(candidate_folders, key=lambda f: "".join(f.split("-")[:2]))

    candidate_checkpoints = glob.glob(
        "*",
        root_dir=os.path.join(logdir_dir, latest_folder, "checkpoints"),
    )
    r = re.compile(r"best_model_\d+_eval_accuracy=\d+\.(\d+)\.pt")
    latest_checkpoint = max(
        candidate_checkpoints, key=lambda f: float(r.search(f).group(1))
    )

    destination_folder = os.path.join(checkpoint_dir, dataset, domain)
    os.makedirs(destination_folder, exist_ok=True)

    old_path = os.path.join(logdir_dir, latest_folder, "checkpoints", latest_checkpoint)
    print(old_path)

    new_name = "".join(latest_folder.split("-")[:2]) + "-" + latest_checkpoint
    new_path = os.path.join(destination_folder, new_name)

    # copy the latest checkpoint file to `checkpoint_dir`
    # os.system("cp " + old_path + " " + new_path)
    shutil.copyfile(old_path, new_path)







