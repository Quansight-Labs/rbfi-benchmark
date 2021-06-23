import numpy as np
import pythran
import scipy
from scipy.interpolate import RBFInterpolator
np.random.seed(0)

num = 150
x = np.linspace(0, 1, num)
y = x[:, None]
image = x + y

# Destroy some values
mask = np.random.random(image.shape) > 0.7
image[mask] = np.nan

valid_mask = ~np.isnan(image)
coords = np.array(np.nonzero(valid_mask)).T
values = image[valid_mask]

it = RBFInterpolator(coords, values)

filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
