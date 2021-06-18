import numpy as np
import pythran, cProfile
import scipy
from scipy.interpolate import RBFInterpolator
np.random.seed(0)

print("Pythran Version: ", pythran.__version__)
print("Scipy Version: ", scipy.__version__)
print("NumPy Version: ", np.__version__)

repeat_freq = 10
num_points = [10, 20, 40, 80, 100]
for num in num_points:
    x = np.linspace(0, 1, num)
    y = x[:, None]
    image = x + y
    print(image.shape)

    # Destroy some values
    mask = np.random.random(image.shape) > 0.7
    image[mask] = np.nan

    valid_mask = ~np.isnan(image)
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]

    pr_init = cProfile.Profile()
    pr_init.enable()
    for _ in range(repeat_freq):
        it = RBFInterpolator(coords, values)
    pr_init.disable()
    pr_init.dump_stats('results/cProfile_' + str(coords.shape[0]) + '_init')

    pr_init = cProfile.Profile()
    pr_init.enable()
    for _ in range(repeat_freq):
        filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    pr_init.disable()
    pr_init.dump_stats('results/cProfile_' + str(coords.shape[0]) + '_evaluate')

    del x, y, image, it, filled

    print("Completed Successfully for %d x %d image."%(num, num))