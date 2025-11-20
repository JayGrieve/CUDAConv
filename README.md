# CUDAConv
Highly-parallelized convolution on CPU+GPU Systems

## Components

- [x] Load images in parallel
        - [ ] For later: Stress test how many we can load at once, determine batch
    - [ ] Convert to distributed memory, have each rank get its own image
        - [ ] To test: Can one cuda kernel fit a whole image? How many images
    - [ ] Have cuda block perform convolution, figure out how to distibute among threads
    - [ ] Test how many images we can put onto GPU at once, get speed, determine streaming feasibility
