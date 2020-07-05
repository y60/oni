import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision import transforms

from model import MLP_ONI, VGG_ONI_Cifer10, VGG16_ONI


def plot_eigenvalues(model, layer_idx):
    weight = model[layer_idx*2].weight
    eigenvalues, _ = torch.eig(weight, eigenvectors=False)
    y = eigenvalues.pow(2).sum(dim=1).detach().cpu().numpy()
    y.sort()
    y = y[::-1]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_ylim(0, max(y)+0.2)
    ax.plot(y)
    return fig


def main(args):
    scaling = not args.no_scaling
    orthinit = not args.no_orthinit

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # log
    log_dir = "runs"
    log_dir += "/" + args.dataset
    if args.prefix is not None:
        log_dir += "/" + args.prefix

    log_dir += "/lr:%f_b:%d_depth:%d_oni:%d"\
        % (args.lr, args.batch_size, args.depth, args.oni_itr)
    if args.oni_itr >= 1 and scaling:
        log_dir += "_scaling"
    if orthinit:
        log_dir += "_orthinit"

    log_writer = SummaryWriter(log_dir, flush_secs=10)

    # dataset & model
    dataset_dir = "~/downloads/datasets/"

    if args.dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2861,), (0.1246,))
        ])
        train_data = FashionMNIST(dataset_dir, train=True,
                                  download=True, transform=transform)
        test_data = FashionMNIST(dataset_dir, train=False,
                                 download=True, transform=transform)
        model = MLP_ONI(28*28, 10, depth=args.depth, oni_itr=args.oni_itr,
                        orthinit=orthinit, scaling=scaling).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = CIFAR10(dataset_dir, train=True,
                             download=True, transform=transform)
        test_data = CIFAR10(dataset_dir, train=False,
                            download=True, transform=transform)
        model = VGG_ONI_Cifer10(
            args.k, args.g, args.oni_itr, orthinit=orthinit).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80, 120], 0.2)
    elif args.dataset == "imgnet":
        pass

    kwargs = {'batch_size': args.batch_size,
              'num_workers': 4,
              'shuffle': True}

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        # train
        model.train()
        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                              total=len(train_loader),
                                              leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logit = model(data)
            loss = F.nll_loss(logit, target)
            loss.backward()
            optimizer.step()

            pred = logit.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % 10 == 0:
                log_writer.add_scalar(
                    "train loss", loss, epoch * len(train_loader) + batch_idx)
        train_accuracy = 100. * correct / (len(train_loader) * args.batch_size)

        # test
        model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logit = model(data)
                test_loss += F.nll_loss(logit, target, reduction='sum').item()

                pred = logit.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            test_accuracy = 100. * correct / \
                (len(test_loader) * args.batch_size)

            global_step = (epoch + 1) * len(train_loader)
            log_writer.add_scalar(
                "test loss", loss, global_step)
            log_writer.add_scalar(
                "train accuracy", train_accuracy, global_step)
            log_writer.add_scalar(
                "test accuracy", test_accuracy, global_step)

        # plot the distribution of eigenvalues
        # of the weight matrix of 5th layer
        if torch.isnan(loss):
            break
        if isinstance(model, MLP_ONI):
            log_writer.add_figure("eigen values of 5th layer", plot_eigenvalues(
                model, args.plot_layer), global_step)

        if scheduler is not None:
            scheduler.step()
    log_writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--oni_itr', type=int, default=5)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--g', type=int, default=3)
    parser.add_argument('--plot_layer', type=int, default=4)
    parser.add_argument('--dataset', type=str, default="fmnist")
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--no_scaling', action="store_true")
    parser.add_argument('--no_orthinit', action="store_true")
    args = parser.parse_args()
    main(args)
