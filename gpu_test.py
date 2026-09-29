import torch
import time

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(10000, 10000).to(device)
y = torch.randn(10000, 10000).to(device)

start = time.time()

z = torch.matmul(x, y)

torch.cuda.synchronize()

end = time.time()

print("Completed")
print("Time:", round(end-start, 2), "seconds")