import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        # get the pretrained DenseNet201 network
#         self.densenet = models.densenet201(pretrained=True)
        self.densenet = torch.load('models/complete_model.pth',
                                   map_location=torch.device('cpu'))

        self.num_ftrs = 1024
        self.num_classes = 2

        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features

        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        # get the classifier of the vgg19
        self.classifier = self.densenet.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        x.requires_grad = True
        # register the hook
        h = x.register_hook(self.activations_hook)

        # don't forget the pooling
        x = self.global_avg_pool(x)
        x = x.view((1, 2208))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


def complete_task(img_path, filename):

    # open image from image path
    img = Image.open(img_path)
    img = img.convert('RGB')

    # convert image to proper normaliztion and form a batch
    img = trans(img)
    img = img.view(-1, 3, 224, 224)

    # initailize a model of Densenet class
    model = DenseNet()

    # set model to evaluation mode
    model.eval()

    # forward pass on model
    log_out = model(img)

    # backwards pass to get gradients
    log_out[0][0].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.get_activations(img).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    #  read image in cv2 fromat
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img[int(img.shape[1]/2)-112:int(img.shape[1]/2) +
              112, int(img.shape[0]/2)-112:int(img.shape[0]/2+112)]

    # create heatmap of image size
    heatmap = cv2.resize(np.float32(heatmap), (img.shape[1], img.shape[0]))

    # noramlize
    heatmap = np.uint8(255 * heatmap)

    # apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    #
    superimposed_img = heatmap * 0.4 + img
    # print(superimposed_img)
    cv2.imwrite(os.path.join(os.getcwd(), 'static',
                             'maps', filename), superimposed_img)

    # find fianl probablity
    prob = np.exp(log_out[0].detach().numpy())

    return prob
