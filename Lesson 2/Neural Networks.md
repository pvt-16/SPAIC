## Identifying text in an image

### Using MNIST database
 Greyscale handwritten digits. Each image is 28x28 pixels
 
 Getting dataset - through **torchvision** package and create training and test datasets 

`from torchvision import datasets` -> use this to download MNIST dataset

After downloading the dataset, you need to load the training data.
using torch, we load the data.
`trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)`

Here, batch_size = 64 => 64 images will be loaded in one iteration from the data loader and passed through our network.

And `shuffle=True` tells it to shuffle the dataset every time we start going through the data loader again.
