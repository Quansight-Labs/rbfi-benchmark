import numpy as np
import pythran, yappi
import scipy, sys
from scipy.interpolate import RBFInterpolator
from tabulate import tabulate
np.random.seed(0)

def find_total_and_avg_time(yappi_stats):
    TOTAL_KEY = 6
    AVG_KEY = 14
    total_time = 0
    avg_time = 0
    for stat_dict in yappi_stats:
        total_time += stat_dict[TOTAL_KEY]
        avg_time += stat_dict[AVG_KEY]
    return total_time, avg_time

print("Pythran Version: ", pythran.__version__)
print("Scipy Version: ", scipy.__version__)
print("NumPy Version: ", np.__version__)
print("Yappi Version: ", '1.3.2')

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
    if len(flags) > 0:
        file_name_prefix = 'results/yappi_' + flags + '_'
    else:
        file_name_prefix = 'results/yappi_'
    file = open(file_name_prefix + str(coords.shape[0]) + '_init', 'w')
    with yappi.run(builtins=True):
        for _ in range(repeat_freq):
            it = RBFInterpolator(coords, values)
    init_stats = yappi.get_func_stats()
    init_stats.print_all(out=file)
    total_time, avg_time = find_total_and_avg_time(init_stats)
    result_init.append(total_time)
    result_init.append(avg_time)
    result_init.append(yappi.get_mem_usage())
    file.write("Memory Usage: " + str(yappi.get_mem_usage()) + "\n")
    file.close()
    yappi.clear_stats()

    file = open(file_name_prefix + str(coords.shape[0]) + '_evaluate', 'w')
    with yappi.run(builtins=True):
        for _ in range(repeat_freq):
            filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    eval_stats = yappi.get_func_stats()
    eval_stats.print_all(out=file)
    total_time, avg_time = find_total_and_avg_time(eval_stats)
    result_eval.append(total_time)
    result_eval.append(avg_time)
    result_eval.append(yappi.get_mem_usage())
    file.write("Memory Usage: " + str(yappi.get_mem_usage()) + "\n")
    file.close()
    yappi.clear_stats()

    del x, y, image, it, filled
    results_init.append(result_init)
    results_eval.append(result_eval)
    print("Completed Successfully for %d x %d image."%(num, num))

print("Yappi Profiling Summary")
print("=======================")
print()
print("Repeat Frequency:", repeat_freq)
print()
print("### Initalising RBFInterpolator\n")
print(tabulate(results_init, headers=['Number Of Data Points', 
                                      'Total Time (s)', 'Time/Call (s)',
                                      'Total Memory Usage (bytes)'], tablefmt='github'), end="\n\n")
print("### Evaluating RBFInterpolator\n")
print(tabulate(results_eval, headers=['Number Of Data Points', 
                                      'Total Time (s)', 'Time/Call (s)',
                                      'Total Memory Usage (bytes)'], tablefmt='github'))