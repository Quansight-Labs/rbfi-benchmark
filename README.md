Yappi Profiling Summary (Without SIMD)
======================================

Repeat Frequency: 10

### Initalising RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |   Total Memory Usage (bytes) |
|-------------------------|------------------|-----------------|------------------------------|
|                    1102 |          1.20806 |        0.120752 |                       137592 |
|                    1764 |          3.99963 |        0.399886 |                       137592 |
|                    2556 |          7.81915 |        0.781839 |                       137592 |
|                    3453 |         13.9313  |        1.39303  |                       137592 |
|                    4495 |         29.3855  |        2.93842  |                       137592 |
|                    5643 |         50.9823  |        5.09807  |                       137592 |

### Evaluating RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |   Total Memory Usage (bytes) |
|-------------------------|------------------|-----------------|------------------------------|
|                    1102 |          1.085   |        0.102994 |                       137464 |
|                    1764 |          2.49604 |        0.241569 |                       137464 |
|                    2556 |          5.15016 |        0.50252  |                       137464 |
|                    3453 |          9.48217 |        0.931425 |                       137464 |
|                    4495 |         15.9362  |        1.56944  |                       137464 |
|                    5643 |         24.9028  |        2.46105  |                       137464 |

Yappi Profiling Summary (With SIMD)
===================================

Repeat Frequency: 10

### Initalising RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |   Total Memory Usage (bytes) |
|-------------------------|------------------|-----------------|------------------------------|
|                    1102 |          1.76125 |        0.176073 |                       137592 |
|                    1764 |          4.74023 |        0.473962 |                       137592 |
|                    2556 |          9.40693 |        0.940613 |                       137592 |
|                    3453 |         18.1898  |        1.81887  |                       137592 |
|                    4495 |         33.8257  |        3.38243  |                       137592 |
|                    5643 |         55.6749  |        5.56733  |                       137592 |

### Evaluating RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |   Total Memory Usage (bytes) |
|-------------------------|------------------|-----------------|------------------------------|
|                    1102 |          1.11245 |        0.105558 |                       137464 |
|                    1764 |          2.52389 |        0.244262 |                       137464 |
|                    2556 |          5.61221 |        0.548289 |                       137464 |
|                    3453 |          9.96587 |        0.977694 |                       137464 |
|                    4495 |         17.2801  |        1.70248  |                       137464 |
|                    5643 |         24.7779  |        2.44902  |                       137464 |

cProfile Profiling Summary (Without SIMD)
=========================================

Repeat Frequency: 10

### Initalising RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |
|-------------------------|------------------|-----------------|
|                    1102 |            0.483 |          0.0483 |
|                    1764 |            1.442 |          0.1442 |
|                    2556 |            3.649 |          0.3649 |
|                    3453 |            5.913 |          0.5913 |
|                    4495 |           11.765 |          1.1765 |
|                    5643 |           24.243 |          2.4243 |

### Evaluating RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |
|-------------------------|------------------|-----------------|
|                    1102 |            0.528 |          0.0528 |
|                    1764 |            1.227 |          0.1227 |
|                    2556 |            2.575 |          0.2575 |
|                    3453 |            4.658 |          0.4658 |
|                    4495 |            7.857 |          0.7857 |
|                    5643 |           12.961 |          1.2961 |

cProfile Profiling Summary (With SIMD)
======================================

Repeat Frequency: 10

### Initalising RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |
|-------------------------|------------------|-----------------|
|                    1102 |            0.621 |          0.0621 |
|                    1764 |            1.897 |          0.1897 |
|                    2556 |            3.444 |          0.3444 |
|                    3453 |            7.152 |          0.7152 |
|                    4495 |           16.952 |          1.6952 |
|                    5643 |           26.608 |          2.6608 |

### Evaluating RBFInterpolator

|   Number Of Data Points |   Total Time (s) |   Time/Call (s) |
|-------------------------|------------------|-----------------|
|                    1102 |            0.53  |          0.053  |
|                    1764 |            1.264 |          0.1264 |
|                    2556 |            2.54  |          0.254  |
|                    3453 |            4.906 |          0.4906 |
|                    4495 |            8.332 |          0.8332 |
|                    5643 |           14.482 |          1.4482 |