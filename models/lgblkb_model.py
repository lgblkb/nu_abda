import torch.nn as nn


def conv_relu(*args, module=nn.Conv2d, **kwargs):
    return module(*args, **kwargs), nn.ReLU()


def get_pipelined_conv_relu(in_channels, resultant_out_channels, strip_ending=False, **kwargs):
    items = list()
    for out_channels in resultant_out_channels:
        items.extend(conv_relu(in_channels, out_channels, **kwargs))
        in_channels = out_channels
    if strip_ending: items = items[:-1]
    return nn.Sequential(*items)


class TheModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.part_1 = get_pipelined_conv_relu(1,
                                              [
                                                  32,
                                                  38,
                                                  44,
                                                  50,
                                                  56,
                                                  62,
                                                  68,
                                                  74,
                                              ],
                                              kernel_size=3)
        self.part_2 = get_pipelined_conv_relu(74,
                                              [
                                                  74,
                                                  66,
                                                  58,
                                                  50,
                                                  42,
                                                  34,
                                                  28,
                                                  1,
                                              ],
                                              kernel_size=3,
                                              module=nn.ConvTranspose2d,
                                              strip_ending=True)
    
    def forward(self, x):
        # Computes the outputs / predictions
        x = self.part_1(x)
        x = self.part_2(x)
        return x


class TheModel2(nn.Module):
    def __init__(self):
        super().__init__()
        
        pass
    
    def forward(self, x):
        pass
