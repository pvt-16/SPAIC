## PySyft  

PySyft is an extension of deep learning libraries. PySyft decouples private data from model training and is used for Federated Learning.

Download PySyft at : https://github.com/OpenMined/PySyft

`import syft`

Create a hook - This is to modify PyTorch API to enable privacy DL methods. 

`hook = syft.TorchHook(torch)`
