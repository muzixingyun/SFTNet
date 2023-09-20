import torch
from torch import nn
import efficient
from torchsummary import summary

'''
    基础模块定义
'''


class VggBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_lay):
        super(VggBlock, self).__init__()
        net = []
        for i in range(num_lay):
            if i == 0:
                net.append(
                    nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                              padding=(0, 1, 1))
                )
            else:
                net.append(
                    nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                              padding=(0, 1, 1))
                )
            net.append(nn.BatchNorm3d(out_ch))
            net.append(nn.ReLU(inplace=True))
        net.append(
            nn.MaxPool3d(kernel_size=(1, 2, 2), padding=(0, 0, 0), stride=(1, 2, 2), ceil_mode=False)
        )
        self.vgg_block = nn.Sequential(*net)

    def forward(self, x):
        return self.vgg_block(x)


class Vgg193d_contour(nn.Module):
    def __init__(self):
        super(Vgg193d_contour, self).__init__()
        self.block1 = VggBlock(in_ch=1, out_ch=64, num_lay=1)
        self.block2 = VggBlock(in_ch=64, out_ch=128, num_lay=1)
        self.block3 = VggBlock(in_ch=128, out_ch=256, num_lay=1)
        self.block4 = VggBlock(in_ch=256, out_ch=512, num_lay=1)
        self.block5 = VggBlock(in_ch=512, out_ch=512, num_lay=1)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        return out1, out2, out3, out4, out5


class Vgg193d_space(nn.Module):
    def __init__(self):
        super(Vgg193d_space, self).__init__()
        self.block1 = VggBlock(in_ch=3, out_ch=64, num_lay=2)
        self.block2 = VggBlock(in_ch=64, out_ch=128, num_lay=2)
        self.block3 = VggBlock(in_ch=128, out_ch=256, num_lay=3)
        self.block4 = VggBlock(in_ch=256, out_ch=512, num_lay=3)
        self.block5 = VggBlock(in_ch=512, out_ch=512, num_lay=3)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        return out5


class Channel_Attention(nn.Module):
    def __init__(self):
        super(Channel_Attention, self).__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        attention = self.sig(torch.mean(x, dim=(2, 3))).unsqueeze(-1).unsqueeze(-1).expand(x.shape)
        return attention


class model_Efficient(nn.Module):
    def __init__(self, in_channels):
        super(model_Efficient, self).__init__()
        self.effect = efficient.efficientnetv2_l()

    def forward(self, x):
        return self.effect(x)


class Vgg193d_contour_space(nn.Module):
    def __init__(self):
        super(Vgg193d_contour_space, self).__init__()
        self.vgg_contour = Vgg193d_contour()
        self.block1 = VggBlock(in_ch=3, out_ch=64, num_lay=2)
        self.block2 = VggBlock(in_ch=64, out_ch=128, num_lay=2)
        self.block3 = VggBlock(in_ch=128, out_ch=256, num_lay=4)
        self.block4 = VggBlock(in_ch=256, out_ch=512, num_lay=4)
        self.block5 = VggBlock(in_ch=512, out_ch=512, num_lay=4)

    def forward(self, x_contour, x_face):
        out1_c, out2_c, out3_c, out4_c, out5_c = self.vgg_contour(x_contour)
        out1 = self.block1(x_face)
        out2 = self.block2(out1 + out1_c)
        out3 = self.block3(out2 + out2_c)
        out4 = self.block4(out3 + out3_c)
        out5 = self.block5(out4 + out4_c)
        out = out5 + out5_c
        return out.permute((0, 2, 1, 3, 4))


'''
    模型组合
'''


class vgg193d_lstm(nn.Module):
    def __init__(self, in_channels):
        super(vgg193d_lstm, self).__init__()
        self.vgg_space = Vgg193d_contour_space()
        self.lstm = nn.LSTM(input_size=512 * 3 * 3, hidden_size=1000, num_layers=3, batch_first=True,
                            bidirectional=False, dropout=0.3)
        self.outLay_lstm = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000 * in_channels, 1000)
        )
        self.TemporalAttention = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 8)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x_contour, x_face):
        out_vgg = self.vgg_space(x_contour, x_face)
        b, t, c, w, h = out_vgg.shape
        out_vgg = out_vgg.reshape((b, t, -1))
        time_attention = torch.mean(out_vgg, dim=2)
        time_attention = self.sig(self.TemporalAttention(time_attention))
        out_vgg = time_attention.unsqueeze(2).expand(out_vgg.shape) * out_vgg
        out, (c, h) = self.lstm(out_vgg)
        # out = time_attention.expand(out.shape) * out
        return self.outLay_lstm(out)


class vgg193d_LSTM_Efficient(nn.Module):
    def __init__(self, in_channels):
        super(vgg193d_LSTM_Efficient, self).__init__()
        self.STNet = vgg193d_lstm(in_channels=in_channels)
        self.FTNet = model_Efficient(in_channels=in_channels)
        self.outLay = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)
        )

    def forward(self, X):
        x_contour, x_face = X[:, 3, :, :, :].unsqueeze(1), X[:, 0:3, :, :, :]
        b, c, t, h, w = x_face.shape
        out_STNet = self.STNet(x_contour, x_face)
        out_FTNet = self.FTNet(x_face.reshape((b, t * c, w, h)))
        print(out_STNet.shape)
        print(out_FTNet.shape)
        return self.outLay(torch.cat((out_STNet, out_FTNet), dim=1))


if __name__ == '__main__':
    a = torch.randn((4, 4, 8, 112, 112))
    net = vgg193d_LSTM_Efficient(in_channels=8)
    b = net(a)
    print(net)
    print(b.shape)
    summary(net, a)
