class Discriminator(nn.Module):
  def __init__(self):
        super().__init__()
        self.inp = nn.Conv2d(6, 16, kernel_size=2, stride=2, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=2),
            nn.Sigmoid()
        )

  def forward(self, generated, real):
      concatenated = torch.cat([generated, real], dim=1)
      x = self.inp(concatenated)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      res = self.conv4(x)
      return res
