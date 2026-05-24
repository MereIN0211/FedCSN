import os
import logging
import torch
import numpy as np
from datetime import datetime

from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def set_for_logger(args):
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{args.preprocess_way}_{time_now}.txt"
    log_filepath = os.path.join(args.log_dir, log_filename)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    if logger.hasHandlers():
        logger.handlers.clear()

    # args FileHandler to save log file
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)



def compute_accuracy(model, dataloader, get_confusion_matrix=False, device=None, multiloader=False):
    model.eval()
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss().to(device)

    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    x, target = x.to(device), target.to(device)
                    target = target.long()

                    out = model(x)
                    loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x, target = x.to(device), target.to(device)
                target = target.long()
                out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        split = 'train' if train else 'test'

        pkl_path = os.path.join(base_path, '{}_{}.pkl'.format(site, split))
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing pkl file: {pkl_path}")

        self.paths, self.labels = np.load(pkl_path, allow_pickle=True)
        
        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])

        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
