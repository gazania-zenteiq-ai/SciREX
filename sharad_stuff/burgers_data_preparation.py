import numpy as np
from data_loader_zenodo import Burgers1p1Ddataloader



input_shape, input_data, output_data = Burgers1p1Ddataloader.data_loader()

print("Input shape:", input_shape)
print("Input data shape:", input_data.shape)
print("Output data shape:", output_data.shape)

