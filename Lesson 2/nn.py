{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled1.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "RuJ8ScSbKWr_",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Import necessary packages\n",
        "\n",
        "%matplotlib inline\n",
        "%config InlineBackend.figure_format = 'retina'\n",
        "\n",
        "import numpy as np\n",
        "import torch\n",
        "\n",
        "import helper\n",
        "\n",
        "import matplotlib.pyplot as plt"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uxylohGVHTnI",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from torchvision import datasets, transforms\n",
        "\n",
        "# Define a transform to normalize the data\n",
        "transform = transforms.Compose([transforms.ToTensor(),\n",
        "                              transforms.Normalize((0.5,), (0.5,)),\n",
        "                              ])\n",
        "\n",
        "# Download and load the training data\n",
        "trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)\n",
        "trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7B9JJGHfH4sV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# to loop through the batch\n",
        "dataiter = iter(trainloader)\n",
        "images, labels = dataiter.next()\n",
        "print(type(images))\n",
        "print(images.shape)\n",
        "print(labels.shape)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "A9eVk7XKL81A",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 169
        },
        "outputId": "c3b86551-f096-45e1-fa04-15465287812a"
      },
      "source": [
        "plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-6-bc2cee0c0ad3>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimages\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msqueeze\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcmap\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'Greys_r'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m;\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'images' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Z3qm9-TSL5gc",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "images[1].flatten().shape"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}