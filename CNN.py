import torch.nn as nn
import matplotlib.pyplot as plt

class IconLocatorCNN(nn.Module):
    def __init__(self):
        super(IconLocatorCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        # For simplicity, we assume the image is resized to 136x136. Adjust this if you change the input size.
        self.fc1 = nn.Linear(in_features=64 * 136 * 136, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        
        # Output layer
        self.out = nn.Linear(in_features=128, out_features=2)  # x and y coordinates
    
    def forward(self, x, return_intermediate=False):
        # Pass data through conv layers
        x1 = nn.ReLU()(self.conv1(x))
        x2 = nn.ReLU()(self.conv2(x1))
        x3 = nn.ReLU()(self.conv3(x2))
    
        # Flatten the data
        x_flat = x3.view(x.size(0), -1)
    
        # Pass data through fully connected layers
        x_fc1 = nn.ReLU()(self.fc1(x_flat))
        x_fc2 = nn.ReLU()(self.fc2(x_fc1))
    
        # Output layer
        x_out = self.out(x_fc2)
    
        if return_intermediate:
            return x_out, [x1, x2, x3, x_fc1, x_fc2]
        else:
            return x_out

    @staticmethod
    def visualize_activations(activation_list):
        for index, activation in enumerate(activation_list):
            # Assuming activations are in the shape (batch_size, channels, height, width)
            # For simplicity, we'll visualize the feature maps of the first image in the batch
            batch_size, num_channels, _, _ = activation.shape
        
            # Create a grid of subplots for the current layer's channels
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
        
            for channel in range(num_channels):
                ax = axes[channel]
                ax.imshow(activation[0][channel].cpu().detach().numpy(), cmap='viridis')
                ax.set_title(f"Layer {index + 1}, Channel {channel + 1}")
                fig.colorbar(ax.imshow(activation[0][channel].cpu().detach().numpy(), cmap='viridis'), ax=ax, orientation='vertical')
        
            # Display all channels of the current layer in a single window
            plt.show()

# Instantiate the model
icon_locator_model = IconLocatorCNN()

# Check the model structure
icon_locator_model
