class fcnn(nn.Module):
    def __init__(self,input_dim,output_dim, hidden_dim):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.Relu()

        self.fc2 = nn.Linear(hidden_dim,output_dim)

    
 

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN).__init__()

        self.conv1 = nn.Conv2d(input_channel= 3, output_chanel = 16, kernel_size = 3, stride= 1, padding= 1)
        self.relu = nn.Relu()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(input_channel= 16, output_chanel = 32, kernel_size = 3, stride= 1, padding= 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()



        self.fc = nn.Linear(32 * 8 * 8, 10)  # Assuming input image size is 32x32

        def forward(self,x):

            out = self.conv1(x)
            out = self.relu(out)
            out = self.maxpool(out)
            out = self.conv2(out)
            out = self.relu(out)
            out = self.maxpool2(out)

            out = self.flatten(out)
            out = self.fc(out)

            return out
        

    class Resnet(nn.Module):
        def __init__(self,channels):
            super().__init__()

            self.conv1 = nn.Conv2d(input_channel = 3, output_channel = 16,kernel_size = 3, padding =1)
            self.bn1 = nn.Batchnorm2d(channels)
            self.relu = nn.Relu()

            self.conv2 = nn.Conv2d(input_channel = 16, output_channel = 32,kernel_size = 3, padding = 1)
            self.bn2 = nn.Batchnorm2d(channels)

        def forward(self,x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            out = out + residual

            out = self.relu(out)

            return out

model = Resnet(channels=16)

dummy_input = torch.randn(8,64)

            