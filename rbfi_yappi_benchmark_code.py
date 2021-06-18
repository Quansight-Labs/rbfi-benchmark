import numpy as np
import pythran, yappi
import scipy, sys
from scipy.interpolate import RBFInterpolator
np.random.seed(0)

print("Pythran Version: ", pythran.__version__)
print("Scipy Version: ", scipy.__version__)
print("NumPy Version: ", np.__version__)
flags = '_'.join(sys.argv[1:])
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
    
    print(coords.shape, values.shape)
    if len(flags) > 0:
        file_name_prefix = 'results/yappi_' + flags + '_'
    else:
        file_name_prefix = 'results/yappi_'
    file = open(file_name_prefix + str(coords.shape[0]) + '_init', 'w')
    with yappi.run(builtins=True):
        for _ in range(repeat_freq):
            it = RBFInterpolator(coords, values)
    yappi.get_func_stats().print_all(out=file)
    file.write("Memory Usage: " + str(yappi.get_mem_usage()) + "\n")
    file.close()
    yappi.clear_stats()

    file = open(file_name_prefix + str(coords.shape[0]) + '_evaluate', 'w')
    with yappi.run(builtins=True):
        for _ in range(repeat_freq):
            filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    yappi.get_func_stats().print_all(out=file)
    file.write("Memory Usage: " + str(yappi.get_mem_usage()) + "\n")
    file.close()

    del x, y, image, it, filled

    print("Completed Successfully for %d x %d image."%(num, num))
