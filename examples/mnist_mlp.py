# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The mnist example using MS-AMP. It is adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py."""

from __future__ import print_function
import argparse
import time
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import msamp
from msamp.operators.activation import Activation
from msamp.operators.loss_fn import Loss_fn
from msamp.common.tensor import TypeCast
from msamp.common.dtype import Dtypes


class Net(nn.Module):
    """The neural network model for mnist."""
    def __init__(self):
        """Constructor."""
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 10)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = Activation.flatten(x, 1)
        x = self.fc1(x)
        x = Activation.relu(x)
        # x = Activation.dropout(x, 0.5)    # todo
        x = self.fc2(x)
        output = Activation.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, loss_list):
    """Train the model with given data loader and optimizer.

    Args:
        args (argparse.Namespace): arguments.
        model (torch.nn.Module): the model to train.
        device (torch.device): the device to train on.
        train_loader (torch.utils.data.DataLoader): the data loader for training.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        epoch (int): the number of epoch to run on data loader.
        loss_list (list): list to store loss values.
    """
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
        loss = Loss_fn.nll_loss(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_list.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    """Test the model on test data set.

    Args:
        model (torch.nn.Module): the model to test.
        device (torch.device): the device to test on.
        test_loader (torch.utils.data.DataLoader): the data loader for testing.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast():
                output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()    # sum up batch loss
            test_loss += Loss_fn.nll_loss(output, target).item()    # sum up batch loss     # todo optional parameter: reduction='mean'
            if output.is_fp8_form:
                output = TypeCast.cast_from_fp8(output.view(dtype=torch.uint8), output.scaling_meta, Dtypes.kfloat16)  
            pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
        )
    )


def main():
    """The main function."""
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)'
    )
    parser.add_argument('--epochs', type=int, default=4, metavar='N', help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False, help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--enable-msamp', action='store_true', default=False, help='enable MS-AMP')
    parser.add_argument('--opt-level', type=str, default='O1', help='MS-AMP optimization level')
    parser.add_argument('--enabling-fp8-activation', action='store_true', default=False, help='enable FP8 activation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.enable_msamp:
        print(f'msamp is enabled, opt_level: {args.opt_level}')
        model, optimizer = msamp.initialize(model, optimizer, opt_level=args.opt_level, enabling_fp8_activation=args.enabling_fp8_activation)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    loss_list = []
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch, loss_list)
        test(model, device, test_loader)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        scheduler.step()

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"\nTraining finished, average epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

    if args.save_model:
        torch.save(model.state_dict(), 'mnist_cnn.pt')

    # 绘制损失曲线图
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel
    plt.savefig('./loss.png')
    
    # 保存loss_list为json格式
    with open('./O2_FP8act_loss_bs256.json', 'w') as f:
        json.dump(loss_list, f)


if __name__ == '__main__':
    main()

# python mnist_mlp.py
# python mnist_mlp.py --enable-msamp --opt-level=O2
# python mnist_mlp.py --enable-msamp --opt-level=O2 --enabling-fp8-activation