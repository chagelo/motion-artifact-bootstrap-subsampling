import torch
import torch.nn as nn
input = torch.randn((1, 256, 256))
model = nn.Conv2d(1, 1, 1, 1)
out = model(input)
kspace = torch.fft.fft2(out)
fs = torch.rand(256)
fs.unsqueeze(dim=-1)
kspace = kspace * fs
print(kspace.shape, kspace.dtype)
origin = torch.abs(torch.fft.ifft2(kspace))
out = model(origin)
ans = torch.sum(out)
ans.backward()