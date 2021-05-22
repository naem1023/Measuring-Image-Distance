import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F

# construct model on cuda if available
use_cuda = torch.cuda.is_available()


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class MIS(nn.Module):
    """Measuring Image Distance Model
    """

    def __init__(self, sub_sampling_ratio=16, width=480, height=640, model_selection='mobile'):
        super(MIS, self).__init__()
        self.sub_sampling_ratio = sub_sampling_ratio
        self.width = width
        self.height = height
        self.model_selection = model_selection

        size = (7, 7)
        fc1_out = 128
        fc2_out = int(fc1_out / 4)

        self.feature_extractor = self.get_feature_extraction()
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(size)
        self.fc1 = nn.Linear(7 * 7 * self.extraction_size + 2, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 1)

        print('make MIS model')

    def rol_pooling(self, output_map):
        output = [self.adaptive_max_pool(out)[0] for out in output_map]

        return output

    def forward(self, sample):
        x = sample[0] # image
        target = sample[1] # target coordinate

        x = self.feature_extractor(x)
        x = self.adaptive_max_pool(x)
        x = x.view(x.size(0), -1)
        targetT = torch.transpose(target, 0, 1)

        x_list = x.tolist()
        targetT_list = targetT.tolist()

        for i in range(x.shape[0]):
            for target in targetT_list[i]:
                x_list[i].append(target)
        x = torch.tensor(x_list).cuda()

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.softplus(x)

        return output

    def get_feature_extraction(self):
        """Return network which produces feature map.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_selection == 'vgg':
            model = torchvision.models.vgg11(pretrained=True).to(device)
            self.extraction_size = 512
        elif self.model_selection == 'mobile':
            model = torchvision.models.mobilenet_v3_small(pretrained=True).to(device)
            self.extraction_size = 48

        features = list(model.features)

        # only collect layers with output feature map size (W, H) < 50
        dummy_img = torch.zeros((1, 3, self.width, self.height)).float()  # test image array

        req_features = []
        output = dummy_img.clone().to(device)

        for feature in features:
            output = feature(output)
            #     print(output.size()) => torch.Size([batch_size, channel, width, height])

            # If size of convolution result is threshold, break.
            if output.size()[2] < self.width // self.sub_sampling_ratio \
                    and output.size()[3] < self.height // self.sub_sampling_ratio:
                break
            req_features.append(feature)

        faster_rcnn_feature_extractor = nn.Sequential(*req_features)
        return faster_rcnn_feature_extractor


def get_model(feature_extract=False):
    # model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model = models.resnet18(pretrained=False)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    # summary(model, input_size=(3,480,640))

    print('hi')
    return model


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size