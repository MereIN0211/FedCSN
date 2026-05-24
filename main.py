import logging
import math
import numpy as np
import random
import argparse
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from nets import AlexNet
from methods import (
    train_local,
    test,
)
import copy
import torch
import torchvision.models as models
from utils import OfficeDataset
from utils import DomainNetDataset

def set_random_seed(seed=None):
    """
    Set random seed for reproducibility.
    If seed is None, a random seed is generated and returned.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"[INFO] No seed provided, using random seed {seed}")
    else:
        print(f"[INFO] Using provided seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def statistic_data(data_loaders):
    """
    Compute the global/client-level mean and standard deviation for each client.
    These statistics are used for the Fallback mechanism and testing phase.
    """
    means = []
    stds = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Start computing statistics on {device}...")

    for data_loader in data_loaders:
        channel_mean_sum = torch.zeros(3).to(device)
        channel_std_sum = torch.zeros(3).to(device)
        n_samples = 0

        for data, _ in tqdm(data_loader, desc="Computing Global Stats", leave=False):
            data = data.to(device)
            N, C, H, W = data.shape

            data = data.view(N, C, -1)

            channel_mean_sum += data.mean(2).sum(0)
            channel_std_sum += data.std(2).sum(0)
            n_samples += N

        if n_samples == 0:
            fallback_mean = torch.tensor([0.485, 0.456, 0.406])
            fallback_std = torch.tensor([0.229, 0.224, 0.225])
            means.append(fallback_mean)
            stds.append(fallback_std)
            continue

        channel_mean = (channel_mean_sum / n_samples).cpu()
        channel_std = (channel_std_sum / n_samples).cpu()

        means.append(channel_mean)
        stds.append(channel_std)

    return means, stds


def statistic_data_per_class(train_loaders, num_classes=10):
    """
    Compute class-wise normalization statistics (mean and variance) for each client.
    This corresponds to the 'Class-wise Statistic Construction' module in FedCSN.
    """
    means_per_client = []
    stds_per_client = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for loader in train_loaders:
        sum_per_class = torch.zeros((num_classes, 3)).to(device)
        sumsq_per_class = torch.zeros((num_classes, 3)).to(device)
        count_per_class = torch.zeros((num_classes,)).to(device)

        has_data = False

        for x, y in tqdm(loader, desc="Computing Class Stats", leave=False):
            has_data = True
            x = x.to(device)
            y = y.to(device).long()

            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)

            sample_mean = x_flat.mean(dim=2)
            sample_var = x_flat.var(dim=2, unbiased=False)
            sample_ex2 = sample_var + sample_mean**2

            sum_per_class.index_add_(0, y, sample_mean)

            sumsq_per_class.index_add_(0, y, sample_ex2)

            batch_counts = torch.bincount(y, minlength=num_classes).float()
            count_per_class += batch_counts

        sum_per_class = sum_per_class.cpu()
        sumsq_per_class = sumsq_per_class.cpu()
        count_per_class = count_per_class.cpu()

        means = torch.zeros((num_classes, 3))
        stds = torch.zeros((num_classes, 3))

        if not has_data:
            for c in range(num_classes):
                means[c] = torch.tensor([0.485, 0.456, 0.406])
                stds[c] = torch.tensor([0.229, 0.224, 0.225])
            means_per_client.append(means)
            stds_per_client.append(stds)
            continue

        for c in range(num_classes):
            if count_per_class[c] > 0:
                mean = sum_per_class[c] / count_per_class[c]
                ex2 = sumsq_per_class[c] / count_per_class[c]
                var = ex2 - mean**2
                var = torch.clamp(var, min=1e-6)
                std = torch.sqrt(var)
            else:
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])

            means[c] = mean
            stds[c] = std

        means_per_client.append(means)
        stds_per_client.append(stds)

    return means_per_client, stds_per_client

def get_class_freqs(train_loaders, num_classes):
    """
    Count the number of samples for each class per client.
    Used for the adaptive dominant-class selection strategy.
    """
    class_freqs = []
    for loader in train_loaders:
        freq = torch.zeros(num_classes)
        for _, y in loader:
            for label in y:
                freq[int(label)] += 1
        class_freqs.append(freq)
    return class_freqs


def fallback_missing_classes(means_topk, stds_topk, client_means, client_stds):
    """
    Fallback mechanism for missing or long-tail categories.
    Replaces NaN values in the statistic pool with the client's global statistics
    to ensure the completeness of the global statistic pool.
    """
    num_clients = len(means_topk)
    num_classes = means_topk[0].shape[0]

    final_means = []
    final_stds = []

    for cid in range(num_clients):
        m = means_topk[cid].clone()
        s = stds_topk[cid].clone()

        missing = torch.isnan(m[:, 0])

        if missing.any():
            fallback_mean = client_means[cid].view(1, 3).repeat(num_classes, 1)
            fallback_std = client_stds[cid].view(1, 3).repeat(num_classes, 1)

            m[missing] = fallback_mean[missing]
            s[missing] = fallback_std[missing]

        final_means.append(m)
        final_stds.append(s)

    return final_means, final_stds

def select_dynamic_classes_per_client(means_per_client, stds_per_client, class_freqs, strategy="max_ratio", **kwargs):
    """
    Adaptive Dominant-class Statistic Selection Strategy with Max-K Guardrail.
    This module adaptively selects reliable statistics based on local data distributions
    to avoid injecting noisy augmentation into the federated training.
    """
    def to_tensor(data):
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], torch.Tensor):
                return torch.stack(data)
            else:
                return torch.tensor(data, dtype=torch.float32)
        return data

    class_freqs = to_tensor(class_freqs)
    means_per_client = to_tensor(means_per_client)
    stds_per_client = to_tensor(stds_per_client)

    num_clients = class_freqs.shape[0]
    num_classes = class_freqs.shape[1]

    MAX_K = kwargs.get("max_k", max(1, int(num_classes * 0.5)))

    masks = torch.zeros_like(class_freqs, dtype=torch.bool)
    
    for cid in range(num_clients):
        freqs_c = class_freqs[cid]
        
        if freqs_c.sum() == 0:
            continue

        selected_indices = []
        
        if strategy == "max_ratio":
            gamma = kwargs.get("gamma", 0.7)
            threshold = freqs_c.max() * gamma
            selected_indices = torch.where(freqs_c >= threshold)[0].tolist()

        if len(selected_indices) > MAX_K:
            selected_freqs = freqs_c[selected_indices]
            _, top_k_relative_idx = torch.topk(selected_freqs, MAX_K)
            selected_indices = [selected_indices[i] for i in top_k_relative_idx.tolist()]

        masks[cid, selected_indices] = True

    means_dyn = torch.full_like(means_per_client, float('nan'))
    stds_dyn = torch.full_like(stds_per_client, float('nan'))
    
    for cid in range(num_clients):
        idx = masks[cid].nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            means_dyn[cid, idx] = means_per_client[cid, idx]
            stds_dyn[cid, idx] = stds_per_client[cid, idx]
            
    return means_dyn, stds_dyn, masks


def prepare_data_officecaltech10(args):
    """
    Prepare Office-Caltech-10 Dataset.
    Aligns the length of all domains by subsetting 50% of the minimum data length.
    """
    data_base_path = args.datadir

    loader_kwargs = (
        {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
        if torch.cuda.is_available()
        else {}
    )

    transform_office = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )

    # amazon
    amazon_trainset = OfficeDataset(
        data_base_path, "amazon", transform=transform_office
    )
    amazon_testset = OfficeDataset(
        data_base_path, "amazon", transform=transform_test, train=False
    )
    # caltech
    caltech_trainset = OfficeDataset(
        data_base_path, "caltech", transform=transform_office
    )
    caltech_testset = OfficeDataset(
        data_base_path, "caltech", transform=transform_test, train=False
    )
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, "dslr", transform=transform_office)
    dslr_testset = OfficeDataset(
        data_base_path, "dslr", transform=transform_test, train=False
    )
    # webcam
    webcam_trainset = OfficeDataset(
        data_base_path, "webcam", transform=transform_office
    )
    webcam_testset = OfficeDataset(
        data_base_path, "webcam", transform=transform_test, train=False
    )

    min_data_len = min(
        len(amazon_trainset),
        len(caltech_trainset),
        len(dslr_trainset),
        len(webcam_trainset),
    )
    val_len = int(min_data_len * 0.4)
    min_data_len = int(min_data_len * 0.5)

    amazon_valset = torch.utils.data.Subset(
        amazon_trainset, list(range(len(amazon_trainset)))[-val_len:]
    )
    amazon_trainset = torch.utils.data.Subset(
        amazon_trainset, list(range(min_data_len))
    )

    caltech_valset = torch.utils.data.Subset(
        caltech_trainset, list(range(len(caltech_trainset)))[-val_len:]
    )
    caltech_trainset = torch.utils.data.Subset(
        caltech_trainset, list(range(min_data_len))
    )

    dslr_valset = torch.utils.data.Subset(
        dslr_trainset, list(range(len(dslr_trainset)))[-val_len:]
    )
    dslr_trainset = torch.utils.data.Subset(dslr_trainset, list(range(min_data_len)))

    webcam_valset = torch.utils.data.Subset(
        webcam_trainset, list(range(len(webcam_trainset)))[-val_len:]
    )
    webcam_trainset = torch.utils.data.Subset(
        webcam_trainset, list(range(min_data_len))
    )

    amazon_train_loader = torch.utils.data.DataLoader(
        amazon_trainset, batch_size=args.batch, shuffle=True, **loader_kwargs
    )
    amazon_val_loader = torch.utils.data.DataLoader(
        amazon_valset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )
    amazon_test_loader = torch.utils.data.DataLoader(
        amazon_testset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )

    caltech_train_loader = torch.utils.data.DataLoader(
        caltech_trainset, batch_size=args.batch, shuffle=True, **loader_kwargs
    )
    caltech_val_loader = torch.utils.data.DataLoader(
        caltech_valset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )
    caltech_test_loader = torch.utils.data.DataLoader(
        caltech_testset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )

    dslr_train_loader = torch.utils.data.DataLoader(
        dslr_trainset, batch_size=args.batch, shuffle=True, **loader_kwargs
    )
    dslr_val_loader = torch.utils.data.DataLoader(
        dslr_valset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )
    dslr_test_loader = torch.utils.data.DataLoader(
        dslr_testset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )

    webcam_train_loader = torch.utils.data.DataLoader(
        webcam_trainset, batch_size=args.batch, shuffle=True, **loader_kwargs
    )
    webcam_val_loader = torch.utils.data.DataLoader(
        webcam_valset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )
    webcam_test_loader = torch.utils.data.DataLoader(
        webcam_testset, batch_size=args.batch, shuffle=False, **loader_kwargs
    )

    train_loaders = [
        amazon_train_loader,
        caltech_train_loader,
        dslr_train_loader,
        webcam_train_loader,
    ]
    val_loaders = [
        amazon_val_loader,
        caltech_val_loader,
        dslr_val_loader,
        webcam_val_loader,
    ]
    test_loaders = [
        amazon_test_loader,
        caltech_test_loader,
        dslr_test_loader,
        webcam_test_loader,
    ]

    amazon_weight = len(amazon_trainset) / (
        len(amazon_trainset)
        + len(caltech_trainset)
        + len(dslr_trainset)
        + len(webcam_trainset)
    )
    caltech_weight = len(caltech_trainset) / (
        len(amazon_trainset)
        + len(caltech_trainset)
        + len(dslr_trainset)
        + len(webcam_trainset)
    )
    dslr_weight = len(dslr_trainset) / (
        len(amazon_trainset)
        + len(caltech_trainset)
        + len(dslr_trainset)
        + len(webcam_trainset)
    )
    webcam_weight = len(webcam_trainset) / (
        len(amazon_trainset)
        + len(caltech_trainset)
        + len(dslr_trainset)
        + len(webcam_trainset)
    )

    return (
        train_loaders,
        val_loaders,
        test_loaders,
        [amazon_weight, caltech_weight, dslr_weight, webcam_weight],
    )


def prepare_data_domainnet(args):
    """
    Prepare DomainNet Dataset.
    Loads data paths and labels dynamically from preprocessed .pkl files.
    """
    if "DomainNet" in args.datadir:
        data_base_path = args.datadir
    else:
        data_base_path = os.path.join(args.datadir, "DomainNet")

    loader_kwargs = (
        {
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
        if torch.cuda.is_available()
        else {}
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    train_loaders = []
    test_loaders = []

    val_loaders = []

    total_train_samples = 0
    train_sample_counts = []

    logging.info(
        f"Loading Full DomainNet ({args.num_classes} classes) from .pkl files..."
    )

    for site in domains:
        trainset = DomainNetDataset(
            data_base_path, site, train=True, transform=transform_train
        )

        testset = DomainNetDataset(
            data_base_path, site, train=False, transform=transform_test
        )

        train_loader = DataLoader(
            trainset, batch_size=args.batch, shuffle=True, **loader_kwargs
        )
        test_loader = DataLoader(
            testset, batch_size=args.batch, shuffle=False, **loader_kwargs
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

        total_train_samples += len(trainset)
        train_sample_counts.append(len(trainset))

        logging.info(f"  Domain [{site}]: Train={len(trainset)}, Test={len(testset)}")

    freqs = [count / total_train_samples for count in train_sample_counts]

    return train_loaders, val_loaders, test_loaders, freqs

def prepare_data_officehome(args):
    """
    Prepare Office-Home Dataset.
    Dynamically loads images from folders and splits them (70% train, 20% test).
    Validation sets are discarded to save memory.
    """
    if "OfficeHome" in args.datadir or "Office-Home" in args.datadir or "OfficeHomeDataset_10072016" in args.datadir:
        data_base_path = args.datadir
    else:
        if os.path.exists(os.path.join(args.datadir, "OfficeHomeDataset_10072016")):
            data_base_path = os.path.join(args.datadir, "OfficeHomeDataset_10072016")
        elif os.path.exists(os.path.join(args.datadir, "OfficeHome")):
            data_base_path = os.path.join(args.datadir, "OfficeHome")
        else:
            data_base_path = os.path.join(args.datadir, "Office-Home")

    loader_kwargs = (
        {"num_workers": 4, "pin_memory": True, "persistent_workers": True}
        if torch.cuda.is_available()
        else {}
    )

    transform_train = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ]
    )

    domains = ["Art", "Clipart", "Product", "Real_World"]
    
    train_loaders = []
    val_loaders = []
    test_loaders = []
    
    total_train_samples = 0
    train_sample_counts = []

    logging.info(f"Loading Office-Home Dataset ({args.num_classes} classes) from {data_base_path}...")

    for site in domains:
        site_path = os.path.join(data_base_path, site)
        if not os.path.exists(site_path):
            raise FileNotFoundError(f"Cannot find Office-Home domain path: {site_path}.")

        full_set_train = datasets.ImageFolder(root=site_path, transform=transform_train)
        full_set_test  = datasets.ImageFolder(root=site_path, transform=transform_test)

        total_len = len(full_set_train)

        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)

        indices = list(range(total_len))

        random.seed(42)
        random.shuffle(indices)

        train_subset = Subset(full_set_train, indices[:train_end])
        val_subset   = Subset(full_set_test,  indices[train_end:val_end])
        test_subset  = Subset(full_set_test,  indices[val_end:])

        train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, **loader_kwargs)
        val_loader   = DataLoader(val_subset,   batch_size=args.batch, shuffle=False, **loader_kwargs)
        test_loader  = DataLoader(test_subset,  batch_size=args.batch, shuffle=False, **loader_kwargs)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

        total_train_samples += len(train_subset)
        train_sample_counts.append(len(train_subset))

        logging.info(f"  Domain [{site}]: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")

    freqs = [count / total_train_samples for count in train_sample_counts]

    return train_loaders, val_loaders, test_loaders, freqs


def get_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Office_Caltech_10",
        help="Target dataset: Office_Caltech_10, DomainNet, OfficeHome",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        required=False,
        default="./data/",
        help="Data directory",
    )
    parser.add_argument(
        "--partition", type=str, default="iid", help="the data partitioning strategy"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The parameter for the dirichlet distribution for data partitioning",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="how many clients are sampled in each round",
    )
    parser.add_argument(
        "--comm_round",
        type=int,
        default=100,
        help="number of maximum communication round",
    )
    parser.add_argument("--epochs", type=int, default=5, help="number of local epochs")
    parser.add_argument("--client_num", type=int, default=4, help="number of clients")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number of classes for classification (default: 10)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to run the program"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="./weights/",
        help="Log directory path",
    )

    parser.add_argument("--optimizer", type=str, default="sgd", help="the optimizer")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="L2 regularization strength"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--model", type=str, default="alexnet", help="Backbone network: alexnet or resnet18"
    )

    args = parser.parse_args()
    return args

def plot_results(results_dict, outdir="results"):
    """
    Plot and save the test accuracy curves over communication rounds.
    """
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = os.path.join(outdir, f"comparison_{timestamp}.png")

    for name, res in results_dict.items():
        rounds = res.get("round", [])
        accs = res.get("acc", [])

        if rounds and rounds[0] != 0:
            rounds = [0] + rounds
            accs = [0.0] + accs

        plt.plot(rounds, accs, label=name)

    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Communication Rounds")
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(outfile)
    # plt.show()

    plt.close()
    logging.info(f"Saved comparison plot to {outfile}")

def run_fedcsn(args):
    """
    Main federated training loop for FedCSN.
    """

    logging.getLogger().handlers.clear()

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    folder_name = f"FedCSN_{args.dataset}_{time_str}"
    log_dir = os.path.join("logs", folder_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"exp_{folder_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )

    logging.info("=== Starting FedCSN Official Implementation ===")
    logging.info("================ Experiment Arguments ================")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"* {arg:<20}: {value}")
    logging.info("======================================================")
    logging.info(f"Target Dataset: {args.dataset}")
    device = torch.device(args.device)

    # ==========================================
    # 1. Data Preparation
    # ==========================================

    if args.dataset == "Office_Caltech_10":
        logging.info(f"Loading Office-Caltech-10 data")
        train_loaders, val_loaders, test_loaders, fed_avg_freqs = prepare_data_officecaltech10(args)
    elif args.dataset == "DomainNet":
        logging.info("Loading DomainNet data...")
        train_loaders, val_loaders, test_loaders, fed_avg_freqs = prepare_data_domainnet(args)
    elif args.dataset == "OfficeHome":
        logging.info("Loading Office-Home data...")
        train_loaders, val_loaders, test_loaders, fed_avg_freqs = prepare_data_officehome(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    client_num = len(train_loaders)
    logging.info(f"Detected {client_num} clients")

    # ==========================================
    # 2. Local Statistics Computation
    # ==========================================
    t_start_stats = time.time()

    logging.info("Computing client-level stats (for fallback & testing) ...")
    means, stds = statistic_data(train_loaders)

    logging.info("Computing per-class stats per client ...")
    means_per_client, stds_per_client = statistic_data_per_class(
        train_loaders, num_classes=args.num_classes
    )

    logging.info("Computing class frequencies for Dynamic Selection ...")
    class_freqs = get_class_freqs(train_loaders, args.num_classes)

    t_end_stats = time.time()
    logging.info(f"All stats prepared. Cost: {t_end_stats - t_start_stats:.2f} seconds")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    if args.model == "alexnet":
        global_model = AlexNet(num_classes=args.num_classes).to(device)
    elif args.model == "resnet18":
        global_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = global_model.fc.in_features
        global_model.fc = torch.nn.Linear(num_ftrs, args.num_classes)
        global_model = global_model.to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    local_models = [
        copy.deepcopy(global_model).to(device) for _ in range(client_num)
    ]
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # ==========================================
    # 4. FedCSN Core: Dynamic Selection & Guardrail
    # ==========================================
    if args.dataset == "Office_Caltech_10":
        absolute_max_k = 5
    elif args.dataset == "DomainNet":
        absolute_max_k = 50
    elif args.dataset == "OfficeHome":
        absolute_max_k = 15
    else:
        absolute_max_k = max(1, int(args.num_classes * 0.5))

    logging.info(f"FedCSN Strategy: Max-Ratio (gamma=0.7), Guardrail Max_K={absolute_max_k}")

    strat_kwargs = {
        "strategy": "max_ratio",
        "gamma": 0.7,
        "max_k": absolute_max_k
    }

    # Perform class-wise selection to filter noisy statistics
    means_topk, stds_topk, masks = select_dynamic_classes_per_client(
        means_per_client, stds_per_client, class_freqs, **strat_kwargs
    )

    selected_counts = masks.sum(dim=1).tolist()
    avg_selected = sum(selected_counts) / len(selected_counts)
    logging.info(f"    -> Client upload breakdown: {selected_counts}")
    logging.info(f"    -> Global Average Uploaded Classes: {avg_selected:.2f}")

    # Fallback Mechanism: Fill missing categories with client-level stats
    means_pc_used, stds_pc_used = fallback_missing_classes(
        means_topk, stds_topk, means, stds
    )

    # ==========================================
    # 5. Federated Training Loop
    # ==========================================
    results = {"round": [], "acc": []}

    # Cosine Annealing Learning Rate scheduling
    eta_max = args.lr
    eta_min = args.lr * 0.1
    T_max = args.comm_round

    for rnd in range(1, args.comm_round + 1):
        current_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * (rnd - 1) / T_max))

        logging.info(
            f"---- Round {rnd} | method=FedCSN | LR={current_lr:.6f} ----"
        )

        t_round_start = time.time()
        t_train_start = time.time()

        # Local Training using Class-wise Selective Normalization
        for cid in range(client_num):
            train_local(
                net_id=cid,
                net=local_models[cid],
                train_dataloader=train_loaders[cid],
                epochs=args.epochs,
                lr=current_lr,
                optimizer_type=args.optimizer,
                weight_decay=args.weight_decay,
                device=device,
                means_per_client=means_pc_used,
                stds_per_client=stds_pc_used,
            )

        # FedAvg: Parameter Aggregation
        global_w = None
        for cid in range(client_num):
            w = local_models[cid].state_dict()
            if global_w is None:
                global_w = {k: w[k] * fed_avg_freqs[cid] for k in w}
            else:
                for k in w:
                    global_w[k] += w[k] * fed_avg_freqs[cid]

        # Distribute updated global model to clients
        global_model.load_state_dict(global_w)
        for i in range(client_num):
            local_models[i].load_state_dict(global_w)

        t_train_end = time.time()
        logging.info(
            f"    [Time] Training & Aggregation: {t_train_end - t_train_start:.2f} s"
        )

        t_test_start = time.time()
        avg_acc = 0
        for cid in range(client_num):
            _, acc = test(
                global_model,
                test_loaders[cid],
                loss_fn,
                device,
                means=[means[cid]],
                stds=[stds[cid]],
            )

            logging.info(f">> Client {cid} Test accuracy: {acc:.4f}")
            avg_acc += acc

        t_test_end = time.time()
        logging.info(f"    [Time] Testing: {t_test_end - t_test_start:.2f} s")

        avg_acc /= client_num
        results["round"].append(rnd)
        results["acc"].append(avg_acc)

        t_round_end = time.time()
        logging.info(f">> Average Test accuracy: {avg_acc:.4f}")
        logging.info(
            f"==== Round {rnd} finished in {t_round_end - t_round_start:.2f} s ===="
        )

    # Save training curves
    logging.info(f"[FedCSN] best acc = {max(results['acc']):.4f}")
    outdir_name = f"results_FedCSN_{args.dataset}_{args.model}"
    plot_results({"FedCSN": results}, outdir=outdir_name)

    return {"FedCSN": results}

if __name__ == "__main__":
    args = get_args()
    seed = set_random_seed(seed=None)
    args.seed = seed

    run_fedcsn(args)
