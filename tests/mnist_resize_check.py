from __future__ import print_function

import argparse
import random
from datetime import datetime, timedelta
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.
    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(epoch, cuda, model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data,
                )
            )


def test(cuda, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
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


# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"

import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = "tight"
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [orig_img] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title="Original image")
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%d%H%M%S%f")[:-5], type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "-c", "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "--epochs", help="number of epochs", default=3, type=int)

    parser.add_argument("--step-size", help="step size", default=0.01, type=float)
    parser.add_argument("--batch-size", help="", default=64, type=int)
    parser.add_argument("--test-batch-size", help="", default=1000, type=int)
    parser.add_argument("--momentum", help="", default=0.5, type=float)
    parser.add_argument("--cuda", help="", default=1, type=int)

    # fmt: on

    args = parser.parse_args()
    set_random_seed(args.seed)

    start = timer()
    # load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../mnist_data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
    )

    orig_img = train_loader.dataset.data[0]
    trans = transforms.ToPILImage()
    img = trans(orig_img)
    img.show()
    resized_imgs = [transforms.Resize(size=14)(train_loader.dataset.data[0])]
    # orig_img = Image.open(Path('assets') / 'astronaut.jpg')
    print("total time: \t", str(timedelta(seconds=timer() - start)))


if __name__ == "__main__":
    main()
