# import fire
import os
import re

def parse_log(log_file_path:str):
    print("")
    print(log_file_path)
    print("-------------------")
    with open(log_file_path) as f:
        txt = f.read()
        re_acc = r"Test accuracy\: (\d\.\d*)"
        re_task = r"\*start test on ([^\*]+)\*"

        acc_list = re.findall(re_acc, txt) 
        task_list = re.findall(re_task, txt)

        for acc, task in zip(acc_list, task_list):
            print(f"{task}:\t{acc}")
    print("-------------------")

def parse_dir(log_dir:str):
    print(f"start parse logs in {log_dir}")
    for path, dir_list, file_list in os.walk(log_dir):
        # for dir_name in dir_list:
        for file in file_list:
            # print(file)
            parse_log(os.path.join(log_dir, file))

if __name__ == "__main__":
    # parse_log("/data/wjn/academic/misc/MultiSourceTrainner/logs/hscore_grad_visda-c/domain_t4_fewshot_10_zero_norm.txt")
    parse_dir("/data/wjn/academic/misc/MultiSourceTrainner/logs/hscore_grad_visda-c/")
    # parse_dir("/data/wjn/academic/misc/MultiSourceTrainner/logs/visda-c_MCW/")