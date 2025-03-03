import torch
import torch.nn.functional as F

#############################################
#           Multi-Layer Perceptron          #
#############################################


def mlp(input_size, layer_sizes, output_size, output_activation=torch.nn.Identity, activation=torch.nn.ELU):
    """
    Create a Multi-Layer Perceptron (MLP) model.

    Args:
        input_size (int): Size of the input layer.
        layer_sizes (list of int): Sizes for the hidden layers.
        output_size (int): Size of the output layer.
        output_activation (callable): Activation function for the output layer.
        activation (callable): Activation function for the hidden layers.

    Returns:
        torch.nn.Sequential: A sequential model containing the defined MLP layers.
    """
    # Build a list of all layer sizes: input -> hidden layers -> output
    sizes = [input_size] + layer_sizes + [output_size]

    layers = []  # Container for the network layers
    # Loop through each layer transition and add a linear layer and an activation layer
    for i in range(len(sizes) - 1):
        # Use the provided activation for hidden layers and output_activation for the last layer
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [
            torch.nn.Linear(sizes[i], sizes[i + 1]),
            act()  # Instantiate the activation
        ]
    return torch.nn.Sequential(*layers)

#############################################
#           Dictionary CPU Conversion      #
#############################################


def dict_to_cpu(dictionary):
    """
    Recursively convert all tensors in a dictionary (or nested dictionaries) to CPU.

    Args:
        dictionary (dict): A dictionary possibly containing tensors or nested dictionaries of tensors.

    Returns:
        dict: A new dictionary with all tensors moved to CPU.
    """
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()  # Move tensor to CPU
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            cpu_dict[key] = dict_to_cpu(value)
        else:
            # Leave non-tensor values unchanged
            cpu_dict[key] = value
    return cpu_dict

#############################################
#          Convolutional Layer Helper      #
#############################################


def conv3x3(in_channels, out_channels, stride=1):
    """
    Create a 3x3 convolutional layer with padding and no bias.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution.

    Returns:
        torch.nn.Conv2d: A 3x3 convolutional layer.
    """
    return torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,  # To preserve spatial dimensions
        bias=False
    )

#############################################
#             Residual Block               #
#############################################


class ResidualBlock(torch.nn.Module):
    """
    Residual Block for deep networks with two 3x3 convolutions.
    """

    def __init__(self, num_channels, stride=1):
        """
        Initialize a Residual Block.

        Args:
            num_channels (int): Number of channels for the convolutions.
            stride (int): Stride for the first convolution.
        """
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels,
                             stride)  # First 3x3 conv layer
        # Batch normalization after conv1
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        # Second 3x3 conv layer
        self.conv2 = conv3x3(num_channels, num_channels)
        # Batch normalization after conv2
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying residual connection and activations.
        """
        # First convolutional block: conv -> batch norm -> ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolutional block: conv -> batch norm
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection: add input tensor to output
        out += x
        # Final activation after addition
        out = F.relu(out)
        return out

#############################################
#             DownSample Module            #
#############################################


class DownSample(torch.nn.Module):
    """
    DownSample module that applies convolutions, residual blocks, and pooling
    to reduce the spatial dimensions of the input.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialize the DownSample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Desired number of output channels after downsampling.
        """
        super().__init__()
        # First convolution reduces spatial dimensions and channels are increased to half of out_channels
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        # Two residual blocks after first convolution
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)])

        # Second convolution to further downsample and increase channels to out_channels
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        # Three residual blocks after second convolution
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)])

        # First pooling layer to further reduce dimensions
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Three more residual blocks after pooling
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)])

        # Final pooling layer for output downsampling
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Forward pass through the downsampling module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled output tensor.
        """
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x

#############################################
#             DownsampleCNN Module         #
#############################################


class DownsampleCNN(torch.nn.Module):
    """
    A CNN module for downsampling that uses sequential convolutional, activation, and pooling layers.
    """

    def __init__(self, in_channels, out_channels, h_w):
        """
        Initialize the DownsampleCNN module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            h_w (tuple): Desired (height, width) output after adaptive pooling.
        """
        super().__init__()
        # Calculate mid_channels as the average of in_channels and out_channels
        mid_channels = (in_channels + out_channels) // 2

        # Feature extraction layers: convolution, activation, and pooling
        self.features = torch.nn.Sequential(
            # First convolution: kernel size is double the height dimension
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # Second convolution
            torch.nn.Conv2d(mid_channels, out_channels,
                            kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Adaptive average pooling to achieve the desired output size
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        """
        Forward pass through the DownsampleCNN module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feature extraction and pooling.
        """
        x = self.features(x)
        x = self.avgpool(x)
        return x

#############################################
#         Support-Scalar Transformations   #
#############################################


def support_to_scalar(logits, support_size):
    """
    Transform a categorical distribution (support) into a scalar value.

    This function decodes a probability distribution over discrete supports into a single scalar,
    then inverts a scaling transformation as described in the paper's appendix.

    Args:
        logits (torch.Tensor): Logits representing the categorical distribution.
        support_size (int): The size of the support (half the range of categories).

    Returns:
        torch.Tensor: Scalar value after decoding and inverting the scaling.
    """
    # Compute probabilities via softmax
    probabilities = torch.softmax(logits, dim=1)

    # Create the support vector ranging from -support_size to support_size
    support = torch.tensor(list(range(-support_size, support_size + 1)))
    support = support.expand(probabilities.shape).float().to(
        device=probabilities.device)

    # Compute the expected value (scalar) from the distribution
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling transformation defined in https://arxiv.org/abs/1805.11593
    scaling_factor = 0.001
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * scaling_factor * (torch.abs(x) + 1 +
         scaling_factor)) - 1) / (2 * scaling_factor)) ** 2 - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar into a categorical distribution with (2 * support_size + 1) categories.

    This function applies a scaling transformation (as defined in the paper's appendix) and then encodes
    the scalar value into a one-hot like vector over the support categories.

    Args:
        x (torch.Tensor): Scalar tensor to be transformed.
        support_size (int): The size of the support (half the range of categories).

    Returns:
        torch.Tensor: A tensor representing the categorical distribution over supports.
    """
    # Reduce the scale as defined in https://arxiv.org/abs/1805.11593
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Clamp x to be within the support range
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor

    # Initialize logits (categorical representation) with zeros
    logits = torch.zeros(x.shape[0], x.shape[1],
                         2 * support_size + 1).to(x.device)
    # Scatter the main probability mass to the floor index
    logits.scatter_(2, (floor + support_size).long().unsqueeze(-1),
                    (1 - prob).unsqueeze(-1))

    # Calculate the index for the upper neighboring category
    indexes = floor + support_size + 1
    # Mask out-of-bound indices and associated probabilities
    prob = prob.masked_fill_(indexes > 2 * support_size, 0.0)
    indexes = indexes.masked_fill_(indexes > 2 * support_size, 0.0)
    # Scatter the remaining probability mass
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
