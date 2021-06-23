import numpy as np
import pythran, cProfile, pstats
import scipy, sys
from scipy.interpolate import RBFInterpolator
from tabulate import tabulate
np.random.seed(0)

print("Pythran Version: ", pythran.__version__)
print("Scipy Version: ", scipy.__version__)
print("NumPy Version: ", np.__version__)

results_init = []
results_eval = []
flags = '_'.join(sys.argv[1:])
repeat_freq = 10
num_points = [40, 50, 60, 70, 80, 90]
for num in num_points:
    result_init = []
    result_eval = []
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

    N = coords.shape[0]
    result_init.append(N)
    result_eval.append(N)
    pr_init = cProfile.Profile()
    pr_init.enable()
    for _ in range(repeat_freq):
        it = RBFInterpolator(coords, values)
    pr_init.disable()
    if len(flags) > 0:
        file_name_prefix = 'results/cProfile_' + flags + '_'
    else:
        file_name_prefix = 'results/cProfile_'
    init_stats = pstats.Stats(pr_init)
    total_time = init_stats.get_stats_profile().total_tt
    result_init.append(total_time)
    result_init.append(total_time/repeat_freq)
    pr_init.dump_stats(file_name_prefix + str(coords.shape[0]) + '_init')

    pr_init = cProfile.Profile()
    pr_init.enable()
    for _ in range(repeat_freq):
        filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    pr_init.disable()
    eval_stats = pstats.Stats(pr_init)
    total_time = eval_stats.get_stats_profile().total_tt
    result_eval.append(total_time)
    result_eval.append(total_time/repeat_freq)
    pr_init.dump_stats(file_name_prefix + str(coords.shape[0]) + '_evaluate')

    del x, y, image, it, filled
    results_init.append(result_init)
    results_eval.append(result_eval)
    print("Completed Successfully for %d x %d image."%(num, num))

print("cProfile Profiling Summary")
print("=======================")
print()
print("Repeat Frequency:", repeat_freq)
print()
print("### Initalising RBFInterpolator\n")
print(tabulate(results_init, headers=['Number Of Data Points', 
                                      'Total Time (s)', 'Time/Call (s)'], tablefmt='github'), end="\n\n")
print("### Evaluating RBFInterpolator\n")
print(tabulate(results_eval, headers=['Number Of Data Points', 
                                      'Total Time (s)', 'Time/Call (s)'], tablefmt='github'))