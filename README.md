# Random walker segmentation for Point Cloud Data

This repository contains a **C++ header-only** implementation and **Python wrapper** of marker-based 3D point cloud segmentation, directly based on the following papers:

- [**Random Walks for Image Segmentation**](https://doi.org/10.1109/TPAMI.2006.233) [[unpaywall]](http://leogrady.net/wp-content/uploads/2017/01/grady2006random.pdf)  
  **Grady, L.**  
  *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 28(11), 1768–1783, 2006.  

- [**Rapid and effective segmentation of 3D models using random walks**](https://doi.org/10.1016/j.cagd.2008.09.007) [[unpaywall]](https://users.cs.cf.ac.uk/Yukun.Lai/papers/cagd09.pdf)  
  **Lai, Yu-Kun; Hu, Shi-Min; Martin, Ralph R.; Rosin, Paul L.**  
  *Computer Aided Geometric Design*, 26(6), 665–679, 2009.

## Installation

### Docker (Recommended)

To build the docker container

```bash
git clone --recursive https://github.com/kasparas-k/random-walker-seg-pointcloud.git
cd random-walker-seg-pointcloud
make build-docker
```

To run the docker container (in an interactive way)
```
make run-docker
```

To stop and clean up the container
```
make clean-docker
```

### System

This package's C++ dependencies are: `Eigen3`, `OpenMP` and `pybind11`. They can be installed via conda:

```bash
conda install -c conda-forge eigen libgomp pybind11
```

The Python package can then be installed locally:
```bash
git clone --recursive https://github.com/kasparas-k/random-walker-seg-pointcloud.git
cd random-walker-seg-pointcloud
pip install -e python/
```

Or directly from github:
```bash
pip install "git+https://github.com/kasparas-k/random-walker-seg-pointcloud.git#subdirectory=python"
```

## Usage

The segmentation markers need to be user-provided, so this algorithm needs to be part of a GUI point cloud processing app's backend, or to consume automatically generated markers from another algorithm.

```python
from pc_rwalker import random_walker_segmentation

# assume an xyz point cloud and markers, a list of point index lists, are available

# by default, (n,) shape numpy array will be returned, where each element is each point's segment id
pointwise_segment_ids = random_walker_segmentation(
    xyz, markers, n_neighbors=15, return_flat=True
)

# alternatively, a list of point index lists can be returned,
# corresponding to the indices assigned to each initial marker list
segment_list = random_walker_segmentation(
    xyz, markers, n_neighbors=15, return_flat=False
)
```

An example of segmenting a bunny point cloud is provided in `examples/segment_bunny.py` with an additional dependency on `laspy`. This code will generate a new point cloud you can inspect in a software of your choice, such as [CloudCompare](https://www.cloudcompare.org/).

```bash
python examples/segment_bunny.py
```
