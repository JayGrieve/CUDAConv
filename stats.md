# PER RANK !

# mpirun -np 1 ./imreader (on 189 frames)
Number of images processed: 189

CPU Times:
  Image Loading:   2.63615 s (0.0139479 s/image)
  Image Resizing:  0.519329 s (0.00274777 s/image)

GPU Times:
  Host to Device:  0.13087 s (0.000692433 s/image)
  Kernel Execution:0.00563232 s (2.98006e-05 s/image)
  Device to Host:  0.0925328 s (0.000489592 s/image)
  Total GPU Time:  0.229035 s (0.00121183 s/image)

Total Time:        3.38451 s

# mpirun -np 2 ./imreader (on 189 frames)
Number of images processed: 95

CPU Times:
  Image Loading:   1.33284 s (0.0140299 s/image)
  Image Resizing:  0.216751 s (0.00228159 s/image)

GPU Times:
  Host to Device:  0.069817 s (0.000734916 s/image)
  Kernel Execution:0.00132461 s (1.39432e-05 s/image)
  Device to Host:  0.036727 s (0.0003866 s/image)
  Total GPU Time:  0.107869 s (0.00113546 s/image)

Total Time:        1.65746 s

# mpirun -np 4 ./imreader (on 189 frames)
Number of images processed: 48

CPU Times:
  Image Loading:   0.669044 s (0.0139384 s/image)
  Image Resizing:  0.145642 s (0.00303421 s/image)

GPU Times:
  Host to Device:  0.0416741 s (0.000868211 s/image)
  Kernel Execution:0.000764704 s (1.59313e-05 s/image)
  Device to Host:  0.0222555 s (0.000463655 s/image)
  Total GPU Time:  0.0646943 s (0.0013478 s/image)

Total Time:        0.87938 s

# mpirun -np 8 ./imreader (on 189 frames)
Number of images processed: 24

CPU Times:
  Image Loading:   0.421843 s (0.0175768 s/image)
  Image Resizing:  0.103341 s (0.00430586 s/image)

GPU Times:
  Host to Device:  0.0149044 s (0.000621016 s/image)
  Kernel Execution:0.000374464 s (1.56027e-05 s/image)
  Device to Host:  0.0091159 s (0.000379829 s/image)
  Total GPU Time:  0.0243948 s (0.00101645 s/image)

Total Time:        0.549579 s


# Against sequential run (on 189 frames)
Number of images processed: 189

CPU Times:
  Image Loading:   0.340585 s (0.00180204 s/image)
  Convolution+Save:4.2606 s (0.0225429 s/image)

Total Time:        4.60119 s

# mpirun -np 1 ./imreader (on 827 frames)
Number of images processed: 827

CPU Times:
  Image Loading:   39.3151 s (0.0475394 s/image)
  Image Resizing:  2.79012 s (0.00337379 s/image)

GPU Times:
  Host to Device:  0.396556 s (0.000479512 s/image)
  Kernel Execution:0.0183932 s (2.22409e-05 s/image)
  Device to Host:  0.342546 s (0.000414203 s/image)
  Total GPU Time:  0.757496 s (0.000915956 s/image)

Total Time:        42.8627 s

# mpirun -np 2 ./imreader (on 827 frames)
Number of images processed: 414

CPU Times:
  Image Loading:   17.9927 s (0.0434606 s/image)
  Image Resizing:  1.42687 s (0.00344655 s/image)

GPU Times:
  Host to Device:  0.213677 s (0.000516128 s/image)
  Kernel Execution:0.00622819 s (1.50439e-05 s/image)
  Device to Host:  0.169171 s (0.000408624 s/image)
  Total GPU Time:  0.389076 s (0.000939797 s/image)

Total Time:        19.8086 s

# mpirun -np 4 ./imreader (on 827 frames)
Number of images processed: 206

CPU Times:
  Image Loading:   8.94003 s (0.0433982 s/image)
  Image Resizing:  0.691307 s (0.00335586 s/image)

GPU Times:
  Host to Device:  0.21815 s (0.00105898 s/image)
  Kernel Execution:0.00314746 s (1.52789e-05 s/image)
  Device to Host:  0.0815913 s (0.000396074 s/image)
  Total GPU Time:  0.302889 s (0.00147034 s/image)

Total Time:        9.93423 s

# mpirun -np 8 ./imreader (on 827 frames)
Number of images processed: 103

CPU Times:
  Image Loading:   4.5191 s (0.0438748 s/image)
  Image Resizing:  0.410768 s (0.00398804 s/image)

GPU Times:
  Host to Device:  0.0924923 s (0.000897983 s/image)
  Kernel Execution:0.00169328 s (1.64396e-05 s/image)
  Device to Host:  0.0513198 s (0.000498251 s/image)
  Total GPU Time:  0.145505 s (0.00141267 s/image)

Total Time:        5.07537 s

# Against sequential run (on 827 frames)
Number of images processed: 827

CPU Times:
  Image Loading:   4.63988 s (0.0056105 s/image)
  Convolution+Save:54.117 s (0.0654378 s/image)

Total Time:        58.7569 s