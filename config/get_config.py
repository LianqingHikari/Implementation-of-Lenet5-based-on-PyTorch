import argparse
import json

json_path = "./config/config.json"


# 这里设计了两种加载的参数的方式，json和命令行
# json中保存默认参数，命令行可以修改部分参数

def parse_args():
    # 加载命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--train_image_path", type=str)
    parser.add_argument("--train_label_path", type=str)
    parser.add_argument("--test_image_path", type=str)
    parser.add_argument("--test_label_path", type=str)


    return parser.parse_args()


def load_json(json_path):
    # 加载默认参数
    with open(json_path, 'r', encoding='UTF-8') as f:
        config = json.load(f)
    return config


def merge_config(default, override):
    # 将默认参数和命令行参数合并起来
    # 这里的default是json，override是命令行
    for key, value in override.__dict__.items():
        if value != None:
            default[key] = value
    return default


def get_merge_config():
    # 获取参数
    args = parse_args()
    json_config = load_json(json_path)
    config = merge_config(json_config, args)
    return config
