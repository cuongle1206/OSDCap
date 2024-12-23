
# To install rbdl without root privilage
This solution requires Conda

## Install dependencies
```
conda install -c omnia eigen3

conda install -c statiskit libboost-dev

pip install numpy cython

export PATH="/path-to-your-venv/include/boost/:$PATH"

export PATH="/path-to-your-venv/osd_venv/include/:$PATH"
```

## Clone the RBDL repository
```
git clone --recursive https://github.com/rbdl/rbdl

cd rbdl && mkdir build && cd build
```

## Build options to get Python binding
```
cmake -D CMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/human-mocap/bin -D RBDL_BUILD_ADDON_URDFREADER=ON -D RBDL_BUILD_PYTHON_WRAPPER=ON -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))") ../

make -j8
```
<!-- optional: rm CMakeCache.txt -->