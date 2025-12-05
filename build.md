Running helper
```bash
source .venv/bin/activate
python3 helpers/extract_frames.py videos/ex.mp4
```

For oscar
```bash
interact -q gpu -t 01:00:00 -m 16g -n 8
```

```bash
module load opencv openmpi cuda
```

```bash
mkdir build && cd build
cmake ..
make
```

```bash
# runs on ../frames to ../output_frames
mpirun -np <number_of_processes> ./main
```
