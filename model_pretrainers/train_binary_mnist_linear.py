from __future__ import print_function
import random
import sys
from datetime import timedelta
from timeit import default_timer as timer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from models.fc_net_linear import FCNetLinear

from FlexibleNN import ExperimentJSON, Database, Metric


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.
    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test(cuda, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            target = target % 2
            data, target = Variable(data), Variable(target)
            output = model(data)

            test_loss += F.mse_loss(
                output, target.type(torch.FloatTensor).unsqueeze(1)
            ).data
            pred = 1 * (output.data > 0.5)
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    exp = ExperimentJSON(sys.argv)

    # fmt: off
    error_table = Metric(
        exp.database_name,
        "error_table",
        ["run", "epoch", "step", "running_acc", "running_err", "test_acc", "n_params"],
        ["int", "int", "int", "real", "real", "real", "int"],
        ["run", "epoch", "step"],
    )
    # fmt: on

    set_random_seed(exp.get_int_param("seed"))
    torch.set_num_threads(1)

    start = timer()
    # load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(14),
                    transforms.Normalize((0.1307,), (0.2801,)),
                ]
            ),
        ),
        batch_size=exp.get_int_param("batch_size"),
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../mnist_data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(14),
                    transforms.Normalize((0.1307,), (0.2801,)),
                ]
            ),
        ),
        batch_size=exp.get_int_param("test_batch_size"),
        shuffle=True,
    )

    model = FCNetLinear()
    if exp.get_int_param("cuda"):
        model.cuda()

    #optimizer = optim.Adam(model.parameters(), lr=exp.get_float_param("step_size"))
    optimizer = optim.SGD(model.parameters(), lr=exp.get_float_param("step_size"))

    step = 0
    running_acc = 0
    running_err = 0
    test_acc = 0
    for epoch in range(1, exp.get_int_param("epochs") + 1):
        model.train()
        error_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            step += 1
            # target is checking whether its even or odd
            target = target % 2
            if exp.get_int_param("cuda"):
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = F.mse_loss(
                output, target.type(torch.FloatTensor).unsqueeze(1)
            )
            loss.backward()
            optimizer.step()

            pred = 1 * (output.data > 0.5)
            correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
            running_acc = 0.995 * running_acc + 0.005 * correct / exp.get_int_param(
                "batch_size"
            )
            running_err = 0.995 * running_err + 0.005 * loss.item()
            if batch_idx % 1000 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRunning Acc: {:.3f}".format(
                        epoch,
                        batch_idx,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.data,
                        running_acc,
                    )
                )
            if step % 1000 == 0:
                error_list.append(
                    [
                        str(exp.get_int_param("run")),
                        str(epoch),
                        str(step),
                        str(running_acc.detach().item()),
                        str(running_err),
                        str(test_acc),
                        str(sum(p.numel() for p in model.parameters())),
                    ]
                )
        test_acc = test(exp.get_int_param("cuda"), model, test_loader).detach().item()
        error_table.add_values(error_list)

    print("total time: \t", str(timedelta(seconds=timer() - start)))

    model.eval()
    model.cpu()
    sample = [x for x, y in test_loader][0]
    if exp.get_int_param("cuda"):
        sample = sample
    traced_script_module = torch.jit.trace(model, sample)
    traced_script_module.save("../trained_models/mnist_binary_linear_pretrained_" + str(exp.get_int_param("seed")) + ".pt")


if __name__ == "__main__":
    main()
