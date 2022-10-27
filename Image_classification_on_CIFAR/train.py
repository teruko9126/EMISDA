import argparse
import os
import shutil
import time
import errno
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from EMISDA import ISDALoss_EM

import resnet
import numpy as np

parser = argparse.ArgumentParser(
    description="Implicit Semantic Data Augmentation (ISDA)"
)
parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    help="dataset (cifar10 [default] or cifar100)",
)

parser.add_argument(
    "--print-freq", "-p", default=10, type=int, help="print frequency (default: 10)"
)

parser.add_argument("--layers", default=32, type=int,
                    help="total number of layers")

parser.add_argument(
    "--droprate", default=0.0, type=float, help="dropout probability (default: 0.0)"
)

parser.add_argument(
    "--checkpoint",
    default="checkpoint",
    type=str,
    metavar="PATH",
    help="path to save checkpoint (default: checkpoint)",
)
parser.add_argument(
    "--resume", default="", type=str, help="path to latest checkpoint (default: none)"
)

parser.add_argument("--name", default="", type=str, help="name of experiment")
parser.add_argument(
    "--no",
    default="1",
    type=str,
    help="index of the experiment (for recording convenience)",
)

parser.add_argument(
    "--lambda_0", default=0.5, type=float, help="hyper-patameter_\lambda for ISDA"
)

parser.add_argument("--cluster", default="3",
                    type=int, help="number of cluster")


# Cosine learning rate
parser.add_argument(
    "--cos_lr",
    dest="cos_lr",
    action="store_true",
    help="whether to use cosine learning rate",
)

parser.set_defaults(cos_lr=False)

args = parser.parse_args()

array_of_bestacc = []


# Configurations adopted for training deep networks.
# (specialized for each type of models)
training_configurations = {
    "resnet": {
        "epochs": 160,
        "batch_size": 128,
        "initial_learning_rate": 0.1,
        "changing_lr": [80, 120],
        "lr_decay_rate": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 1e-4,
    },
}


record_path = (
    "./ISDA test/"
    + str(args.dataset)
    + "-"
    + str(args.layers)
    + "_"
    + str(args.name)
    + "/"
    + "no_"
    + str(args.no)
    + "_lambda_0_"
    + str(args.lambda_0)
    + ("_dropout_" if args.droprate > 0 else "")
    + ("_cos-lr_" if args.cos_lr else "")
)

record_file = record_path + "/training_process.txt"
accuracy_file = record_path + "/accuracy_epoch.txt"
loss_file = record_path + "/loss_epoch.txt"
check_point = os.path.join(record_path, args.checkpoint)


def main():

    global best_prec1
    best_prec1 = 0

    global val_acc
    val_acc = []

    global class_num

    class_num = 10

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4),
                                mode="reflect").squeeze()
            ),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    kwargs = {"num_workers": 0, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](
            "../data", train=True, download=True, transform=transform_train
        ),
        batch_size=training_configurations["resnet"]["batch_size"],
        shuffle=True,
        **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](
            "../data", train=False, transform=transform_test
        ),
        batch_size=training_configurations["resnet"]["batch_size"],
        shuffle=True,
        **kwargs
    )

    model = eval("resnet.resnet" + str(args.layers) + "_cifar")(
        dropout_rate=args.droprate
    )

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    fc = Full_layer(int(model.feature_num), class_num)

    print("Number of final features: {}".format(int(model.feature_num)))

    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
            + sum([p.data.nelement() for p in fc.parameters()])
        )
    )

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    isda_criterion = ISDALoss_EM(
        int(model.feature_num), class_num, args.cluster).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": fc.parameters()}],
        lr=training_configurations["resnet"]["initial_learning_rate"],
        momentum=training_configurations["resnet"]["momentum"],
        nesterov=training_configurations["resnet"]["nesterov"],
        weight_decay=training_configurations["resnet"]["weight_decay"],
    )

    model = torch.nn.DataParallel(model).cuda()
    fc = nn.DataParallel(fc).cuda()

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        fc.load_state_dict(checkpoint["fc"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        isda_criterion = checkpoint["isda_criterion"]
        val_acc = checkpoint["val_acc"]
        best_prec1 = checkpoint["best_acc"]
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    # この関数の中でtrainを回している
    for epoch in range(start_epoch, training_configurations["resnet"]["epochs"]):

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train(train_loader, model, fc, isda_criterion, optimizer, epoch)

        print(isda_criterion.estimator.pi)
        isda_criterion.adjust_em(model, train_loader, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, fc, ce_criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "fc": fc.state_dict(),
                "best_acc": best_prec1,
                "optimizer": optimizer.state_dict(),
                "isda_criterion": isda_criterion,
                "val_acc": val_acc,
            },
            is_best,
            checkpoint=check_point,
        )
        print("Best accuracy: ", best_prec1)
        array_of_bestacc.append(best_prec1)
        np.savetxt(accuracy_file, np.array(val_acc))

    print("Best accuracy: ", best_prec1)
    array_of_bestacc.append(best_prec1)
    print(array_of_bestacc)
    print("Average accuracy", sum(val_acc[len(val_acc) - 10:]) / 10)
    # val_acc.append(sum(val_acc[len(val_acc) - 10:]) / 10)
    # np.savetxt(val_acc, np.array(val_acc))
    np.savetxt(accuracy_file, np.array(val_acc))


def train(train_loader, model, fc, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    ratio = args.lambda_0 * \
        (epoch / (training_configurations["resnet"]["epochs"]))

    # switch to train mode
    model.train()
    fc.train()

    end = time.time()

    for i, (x, target) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        # compute output
        loss, output = criterion(model, fc, input_var, target_var, ratio)

        # ? なぜlossとoutputを出力している？片方だけでいいのでは？
        # * outputとtargetを比較して正答率を計算するために使っている
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file, "a+")
            string = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t"
                "Loss {loss.value:.4f} ({loss.ave:.4f})\t"
                "Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t".format(
                    epoch,
                    i + 1,
                    train_batches_num,
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                )
            )

            print(string)
            fd.write(string + "\n")
            fd.close()


def validate(val_loader, model, fc, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()
    fc.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            features = model(input_var)
            output = fc(features)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            fd = open(record_file, "a+")
            string = (
                "Test: [{0}][{1}/{2}]\t"
                "Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t"
                "Loss {loss.value:.4f} ({loss.ave:.4f})\t"
                "Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t".format(
                    epoch,
                    (i + 1),
                    train_batches_num,
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                )
            )
            print(string)
            fd.write(string + "\n")
            fd.close()

    fd = open(record_file, "a+")
    string = (
        "Test: [{0}][{1}/{2}]\t"
        "Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t"
        "Loss {loss.value:.4f} ({loss.ave:.4f})\t"
        "Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t".format(
            epoch,
            (i + 1),
            train_batches_num,
            batch_time=batch_time,
            loss=losses,
            top1=top1,
        )
    )
    print(string)
    fd.write(string + "\n")
    fd.close()
    val_acc.append(top1.ave)

    return top1.ave


class Full_layer(torch.nn.Module):
    """explicitly define the full connected layer"""

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, "model_best.pth.tar"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations["resnet"]["changing_lr"]:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= training_configurations["resnet"]["lr_decay_rate"]

    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = (
                0.5
                * training_configurations["resnet"]["initial_learning_rate"]
                * (
                    1
                    + math.cos(
                        math.pi * epoch /
                        training_configurations["resnet"]["epochs"]
                    )
                )
            )


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
