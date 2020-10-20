import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model.inference import cnn, train, test

def main():
    parser = argparse.ArgumentParser(description='Pytorch learning on Cifar10')
    parser.add_argument('--no-cuda', action='store_true', default=False,
    	                 help='disables cuda')
    parser.add_argument('--learning-rate', type=float, default=0.01,
    	                 help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7,
    	                 help='gamma for scheduler (default: 0.7)')
    parser.add_argument('--epoch', type=int, default=10,
    	                 help='Set the number of epochs (default: 10')
    parser.add_argument('--batch-size', type=int, default=16,
    	                 help='Set the batch size (default: 16)')



    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    dataset1 = datasets.CIFAR10('./data', train=True, download=True,
    	                         transform=transform)
    dataset2 = datasets.CIFAR10('./data', train=False, download=True,
    	                         transform=transform)

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
    	kwargs.update({'num_workers':1,
    		           'pin_memory':True,
    		           'shuffle':True})

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)
    for epoch in range(1, args.epoch + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(test_loader, device, model)
        scheduler.step()

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(16):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
        main()







    