import torch
print(torch.__version__)
print(torch.cuda.is_available())
cuda = torch.device('cuda')
print(torch.cuda.is_available())