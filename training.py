"""
the train of model and calculate of loss and accuracy
"""
import torch
from model.ResNet50 import Bottleneck
from model.ResNet50 import ResNet50
from data.transform import val_loader
import torch.nn.functional as F
import sklearn.metrics as skm
import numpy as np
import torchvision as tv
import os

torch.autograd.set_detect_anomaly(True)
# instantiate some parameters, optimizer and scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 100
num_classes = 200
model = tv.models.resnet50(num_classes=num_classes).to(DEVICE)
# model = ResNet50(num_classes=num_classes)
# model = ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# define the training loop
best_snapshot_path = None
val_acc_avg = list()
best_val_acc = -1.0


def train():
    for epoch in range(num_epochs):

        # model training
        model.train()
        train_loss = list()
        for batch in val_loader:
            x, y = batch

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            # predict bird species
            y_pred = model(x)

            # calculate the loss
            loss = F.cross_entropy(y_pred, y)

            # backprop & update weights
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss.append(loss.item())

        # validate the model
        model.eval()
        val_loss = list()
        val_acc = list()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                # predict bird species
                y_pred = model(x)

                # calculate the loss
                loss = F.cross_entropy(y_pred, y)

                # calculate the accuracy
                acc = skm.accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])

                val_loss.append(loss.item())
                val_acc.append(acc)

            val_acc_avg.append(np.mean(val_acc))

            # save the best model snapshot
            current_val_acc = val_acc_avg[-1]
            if current_val_acc > best_val_acc:
                if best_snapshot_path is not None:
                    os.remove(best_snapshot_path)

        # adjust the learning rate
        scheduler.step()

        # print performance metrics
        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            print('Epoch {} |> Train. loss: {:.4f} | Val. loss: {:.4f}'.format(
                epoch + 1, np.mean(train_loss), np.mean(val_loss)
            ))

train()
