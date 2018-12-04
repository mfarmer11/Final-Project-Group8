# coding: utf-8

import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
import itertools

from torchvision import models

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(1122)
# ------------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# ------------------------------------------------------------------------------------
transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

batch_size = 64

data_path = 'Dataset/Train/'
train_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transforms
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=16

                                           )

data_path = 'Dataset/Test/'
test_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transforms
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=16
                                          )

classes = ('bags', 'dresses', 'footwear', 'outerwear', 'skirts', 'tops')

train_iter = iter(train_loader)
print(type(train_iter))

images, labels = train_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
plt.show()

num_epochs = 10
learning_rate = 0.01


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(14 * 14 * 32, 1000)
        self.fc1 = nn.Linear(1000, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        return out


cnn = CNN()
cnn.cuda()

# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model
start_time = time.clock()
all_train_accuracy = []
total = 0
correct = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

        # Mine - Accuracy for training set
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()

    all_train_accuracy.append(correct.cpu().numpy() / total)
# -----------------------------------------------------------------------------------
print('Training set Accuracy of the model on the Train images: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
# Plot the accuracy for the num of epochs verus accuracy

plt.plot(range(num_epochs), all_train_accuracy, 'g--')
plt.title('Number of Epochs VS accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).to(device)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.to(device)).sum()
end_time = time.clock()
print(str(end_time - start_time) + " seconds")
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
#


class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))

true_label = []
predicted_label = []

for data in test_loader:
    images, labels = data
    images = Variable(images).to(device)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)

    # mine
    y_proba, predicted = torch.max(outputs.data, 1)
    # end mine

    c = (predicted.cpu() == labels)

    # Add true labels

    true_label.extend(labels.cpu().numpy())
    # Add predicted labels
    predicted_label.extend(predicted.cpu().numpy())

    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(6):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------

# cnf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())

# Change later
cnf_matrix = confusion_matrix(true_label, predicted_label)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Confusion matrix')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10, 7))
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix')

plt.show()

target_names = classes
print(classification_report(true_label, predicted_label, target_names=target_names))

#
# # The accuracy rate
# print('Accuracy Rate')
# accuracy_rate = np.sum(np.diagonal(cnf_matrix)) / np.sum(cnf_matrix)
# print(accuracy_rate)
#
# print()
# # The misclassifcation rate
# print('Miscalculation Rate')
# print(1 - (np.sum(np.diagonal(cnf_matrix)) / np.sum(cnf_matrix)))

# ROC and AUC


from sklearn.metrics import roc_curve, auc

# change the class to stop at 5
from sklearn.preprocessing import label_binarize

y_label = label_binarize(true_label, classes=[0, 1, 2, 3, 4, 5])

y_predict = label_binarize(predicted_label, classes=[0, 1, 2, 3, 4, 5])

n_classes = len(classes)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_predict[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# colors = cycle(['blue', 'red', 'green'])


print(roc_auc)

plt.figure(figsize=(15, 10))

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0}  ----- (area = {1:0.2f})'
                   ''.format(classes[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend()
plt.show()

# Test the model


# In[18]:


size = 4
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=size,
                                          shuffle=False, num_workers=4
                                          )

# In[19]:


dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(size)))
plt.show()

# In[20]:

images = Variable(images).to(device)
outputs = cnn(images)

# In[21]:


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(size)))

