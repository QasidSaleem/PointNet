import os
import pickle
import argparse

from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torchextractor as tx
from joblib import dump

from utils.model import pointnet_model


def load_pickle(file_path):
    """"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


class PointNetFeatureExtractor:
    def __init__(
        self, model_path, module_names=["fc2"], device="cuda", num_classes=10
    ):
        pointnet, optimizer, epochs_trained = pointnet_model(
            num_classes=num_classes,
            device=device,
            mode="test",
            train_from="checkpoint",
            model=model_path,
        )
        pointnet.eval()
        self.device = device
        self.model_extractor = tx.Extractor(pointnet, module_names)
        self.model_extractor.eval()

    def extract_feature(self, _data):
        # _data = get_single_data(path)
        with torch.no_grad():
            _input = _data.float().to(self.device)
            _input = torch.unsqueeze(_input, 0)
            model_output, features = self.model_extractor(
                _input.transpose(1, 2)
            )

        return features["fc2"].cpu().numpy()

    def collect_features_all_dataset(self, basedir, filepath):
        path = os.path.join(basedir, filepath)
        features = []
        # with open(path) as f:
        #     lines = f.readlines()
        # for line in tqdm(lines):
        #     mesh_path = os.path.join(basedir, line).strip()
        #     features.append(self.extract_feature(mesh_path))

        data = load_pickle(path)
        for i in range(len(data)):
            features.append(self.extract_feature(data[i].points_list()[0]))
        return np.concatenate(features)


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./Data",
        help="data directory",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="validation_pointclouds.pickle",
        help="input file name",
    )
    parser.add_argument(
        "--out_file_name",
        type=str,
        default="validation_features.joblib",
        help="output file name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./pretrained/pretrained-modelnet10.pth",
        help="model_path",
    )

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = vars(args)

    pointnet_extractor = PointNetFeatureExtractor(
        model_path=args["model_path"], module_names=["fc2"]
    )

    train_features = pointnet_extractor.collect_features_all_dataset(
        args["data_dir"], args["file_name"]
    )
    dump(train_features, os.path.join(args["data_dir"], args["out_file_name"]))


if __name__ == "__main__":
    main()
