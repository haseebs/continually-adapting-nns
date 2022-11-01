from __future__ import print_function

import argparse
from datetime import datetime, timedelta
from timeit import default_timer as timer

import torch
from torchvision import datasets, transforms


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%d%H%M%S%f")[:-5], type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "-c", "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "--epochs", help="number of epochs", default=3, type=int)

    parser.add_argument("--step-size", help="step size", default=0.5, type=float)
    parser.add_argument("--batch-size", help="", default=64, type=int)
    parser.add_argument("--test-batch-size", help="", default=1000, type=int)
    parser.add_argument("--momentum", help="", default=0.5, type=float)
    parser.add_argument("--cuda", help="", default=0, type=int)

    # fmt: on

    args = parser.parse_args()

    # load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../mnist_data",
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
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = torch.jit.load("trained_models/mnist_small.pt")
    if args.cuda:
        model.cuda()

    model.eval()
    sample = [x for x, y in train_loader][0][0]
    if args.cuda:
        sample = sample

    start = timer()
    print("input: ", sample)
    print("pred: ", model(sample).reshape(-1).detach().numpy())
    print("\n\n\n")

    sample = torch.ones(1, 1, 14, 14).type(torch.FloatTensor)
    print("input: ", sample)
    print("pred: ", model(sample).reshape(-1).detach().numpy())
    print("\n\n\n")

    print("weights: ")
    for k in model.named_parameters():
        print(k)

    print("time for preds: \t", str(timedelta(seconds=timer() - start)))
    print("fast af...")


if __name__ == "__main__":
    main()
