import numpy as np

# Used to map colors of mask from PascalVOC to class index
label_colors = np.array(
    [
        (0, 0, 0),  # Class 0
        (192, 128, 128),  # Class  1
        (192, 0, 0),  # Class  2
        (0, 64, 128),  # Class  3
        (64, 0, 0),  # Class  4
        (128, 64, 0),  # Class 5
        (0, 192, 0),  # Class   6
        (128, 128, 0),  # Class  7
        (128, 128, 128),  # Class 8
        (0, 128, 0),  # Class  9
        (0, 0, 128),  # Class   10
        (128, 192, 0),  # Class   11
        (64, 128, 0),  # Class 12
        (192, 0, 128),  # Class  13
        (64, 0, 128),  # Class   14
        (128, 0, 0),  # Class  15
        (0, 128, 128),  # Class  16
        (0, 64, 0),  # Class  17
        (64, 128, 128),  # Class 18
        (192, 128, 0),  # Class  19
        (128, 0, 128),  # Class  20
    ]
)
