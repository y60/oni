import torch
import torch.nn as nn
from oni_module import Linear_ONI, Conv2d_ONI


class MLP_ONI(nn.Sequential):
    def __init__(self, in_c, out_c, f_c=256, depth=10, oni_itr=5,
                 orthinit=True, scaling=True):
        assert depth >= 2
        layers = []
        layers.append(Linear_ONI(
            in_c, f_c, oni_itr=oni_itr, orthinit=orthinit, scaling=scaling))
        layers.append(nn.ReLU())
        for i in range(1, depth-1):
            layers.append(Linear_ONI(
                f_c, f_c, oni_itr=oni_itr, orthinit=orthinit, scaling=scaling))
            layers.append(nn.ReLU())
        layers.append(Linear_ONI(
            f_c, out_c, oni_itr=oni_itr, orthinit=orthinit))
        layers.append(nn.LogSoftmax(dim=1))
        super().__init__(*layers)


class VGG_ONI(nn.Module):
    # Modification of torchvision.models.vgg
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    def __init__(self, cfg, batch_norm=False, oni_itr=5,
                 init_weights=True, num_classes=1000, gap=False,
                 orthinit=True):
        super().__init__()
        self.features = self._make_layers(cfg, batch_norm, oni_itr, orthinit)
        last_c = 0
        for v in cfg[::-1]:
            if isinstance(v, int):
                last_c = v
                break
            elif isinstance(v, tuple):
                last_c = v[0]
                break
        if gap:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.classifier = self._make_classifier(
                last_c * 1 * 1, num_classes)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = self._make_classifier(
                last_c * 7 * 7, num_classes)
        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg, batch_norm=False, oni_itr=5, orthinit=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if isinstance(v, tuple):
                    v, stride = v
                else:
                    stride = 1
                if oni_itr > 0 or orthinit:
                    conv2d = Conv2d_ONI(
                        in_channels, v, kernel_size=3, stride=stride,
                        padding=1, oni_itr=oni_itr, orthinit=orthinit)
                else:
                    conv2d = nn.Conv2d(
                        in_channels, v, kernel_size=3, stride=stride,
                        padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _make_classifier(self, in_c, num_classes):
        return nn.Sequential(
            nn.Linear(in_c, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TODO
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG_ONI_Cifer10(VGG_ONI):
    def __init__(self, k, g, oni_itr=5, orthinit=True):
        cfg = [32 * k]
        cfg += [32 * k] + [32 * k] * (g-1)
        cfg += [(64 * k, 2)] + [64 * k] * (g-1)
        cfg += [(128 * k, 2)] + [128 * k] * (g-1)
        super().__init__(cfg, num_classes=10, oni_itr=oni_itr,
                         orthinit=orthinit)


class VGG16_ONI(VGG_ONI):
    def __init__(self, num_classes=1000):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
               'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        super().__init__(cfg, num_classes=num_classes)
