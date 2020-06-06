# модель похожа на эту https://iopscience.iop.org/article/10.1088/1742-6596/1345/5/052066/pdf но с меньшим числом dilated сверток (т.к. размер картинки меньше)
class DilatedUNet_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)

        )
        self.pool0 = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1, dilation=2) # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 32,
                      out_channels = 48,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 48,
                      out_channels = 64,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.pool1 = nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=1, dilation=2)# 128 -> 64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                      out_channels = 96,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 96,
                      out_channels = 128,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

        self.pool2 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=1, dilation=2) # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 128,
                      out_channels = 176,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(176),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 176,
                      out_channels = 256,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.pool3 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=1, dilation=2) # 32 -> 16

        # dilation
        self.dilated_conv0 = nn.Sequential(
            nn.Conv2d(256, out_channels=256, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.dilated_conv1 =  nn.Sequential(
            nn.Conv2d(256, out_channels=256, kernel_size=3, dilation=2,  padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(256, out_channels=256, kernel_size=3, dilation=4,  padding=4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(256, out_channels=256, kernel_size=3, dilation=8,  padding=8),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d( 256, 256, kernel_size = 2, stride = 2) # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels = 512,
                      out_channels = 256,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 256,
                      out_channels = 128,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample1 = nn.ConvTranspose2d(128, 128, kernel_size = 2, stride = 2,  padding=1, dilation=2, output_padding=1)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 256,
                      out_channels = 128,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 128,
                      out_channels = 64,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample2 = nn.ConvTranspose2d( 64, 64, kernel_size = 2, stride = 2,  padding=1, dilation=2, output_padding=1)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 128,
                      out_channels = 64,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 64,
                      out_channels = 32,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample3 = nn.ConvTranspose2d(32, 32, kernel_size = 2, stride = 2,  padding=1, dilation=2, output_padding=1) # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,
                      out_channels = 32,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 32,
                      out_channels = 32,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels = 32,
                      out_channels = 16,
                      kernel_size = 3,
                      padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels = 16,
                      out_channels = 3,
                      kernel_size = 3,
                      padding = 1)

        )

    def forward(self, x):
        # encoder
        e0 = self.pool0(self.enc_conv0(x))
        e1 = self.pool1(self.enc_conv1(e0))
        e2 = self.pool2(self.enc_conv2(e1))
        e3 = self.pool3(self.enc_conv3(e2))
        # bottleneck
        dilated0 = self.dilated_conv0(e3)
        dilated1 = self.dilated_conv1(dilated0)
        dilated2 = self.dilated_conv2(dilated1)
        dilated3 = self.dilated_conv3(dilated2)

        b = dilated0 + dilated1 + dilated2 + dilated3

        # decoder
        d0 = self.dec_conv0(torch.cat((self.upsample0(b), self.enc_conv3(e2)), dim=1))
        d1 = self.dec_conv1(torch.cat((self.upsample1(d0), self.enc_conv2(e1)), dim=1))
        d2 = self.dec_conv2(torch.cat((self.upsample2(d1), self.enc_conv1(e0)), dim=1))
        d3 = self.dec_conv3(torch.cat((self.upsample3(d2), self.enc_conv0(x)), dim=1)) # no activation


        return torch.tanh(d3)
