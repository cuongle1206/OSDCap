
<div align="center">
<img src="Figures/Teaser.png" width="800" alt="logo"/>
</div>

# Official Implementation of OSDCap

Optimal-state Dynamics Estimation for Physics-based Human Motion Capture from Videos

[![Paper](https://img.shields.io/badge/arXiv-2410.07795-red)](https://arxiv.org/abs/2410.07795)
... Inprogress ...

## Dependencies

- Miniconda 23.5.2
- Python 3.8
- RBDL 3.3.1
- TRACE 

## Installation

Follow the instruction from [TRACE](https://github.com/Arthur151/ROMP/tree/master/simple_romp/trace2) to install and extract the initial kinematics estimations from input videos. We recommend create a separate Conda environment to do this. Otherwise, the pre-extracted kinematics from TRACE can be downloaded from [here](https://liuonline-my.sharepoint.com/:u:/g/personal/cuole74_liu_se/EbOH95Kh4-VEoLNNPfJaanwBv_CTj8wu99iKR4ZFidVChQ?e=SufmDw).

Build and install from source with Python binding from [RBDL](https://github.com/rbdl/rbdl). If you don't have root-privilege (such as when working on remote server), please refer to this [instruction](RBDL_install.md). 

To install OSDCap's dependencies

```
pip install -r requirements.txt
```

## Experiments

### Extraction of Human 3.6M ground truth
Generate ground truth data for Human 3.6M by transforming them to friendlier format. Please log in and download the annotation of Human 3.6M from the official [website](http://vision.imar.ro/human3.6m/description.php). We based our extracting and processing code on [h36m-fetch](https://github.com/anibali/h36m-fetch).

Your h36m directory should look similar to this after the extraction:

```
|-- extracted
|   |-- S1
|   |   |-- Poses_D2_Positions
|   |   |-- Poses_D3_Angles
|   |   |-- Poses_D2_Angles_mono
|   |   |-- Poses_D3_Positions
|   |   |-- Poses_D3_Positions_mono
|   |   |-- Poses_D3_Positions_mono_universal
|   |   |-- Poses_RawAngles
|   |   |-- Videos
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```

The processed data will locate in datasets/h36m/processed/
```
cd datasets/h36m/
python process_extracted.py -p "your-h36m-directory"
cd ../..
```

### Generation of training and testing database for OSDCap

Please put the extracted kinematics from TRACE as following:
```
|-- datasets
|   |-- h36m
|   |   |-- TRACE_results
|   |-- fit3d
|   |   |-- TRACE_results
|   |-- sport
|   |   |-- TRACE_results
```
But of course you can put them anywhere that is convienient to you and change the path in [here](data_gen.py#15).


To generate the training and testing database for OSDCap, run data_gen.py
```
python data_gen.py -dst h36m
python data_gen.py -dst fit3d
python data_gen.py -dst sport
```


To train the networks on a specific dataset
```
python main.py -trn -dst h36m
```

To test the trained models on a specific dataset
```
python main.py -dst h36m
```

## Visualization


## Citation
If you find our work helpful, please cite the paper as
```bibtex
@article{le2024_osdcap,
  title = {Optimal-State Dynamics Estimation for Physics-based Human Motion Capture from Videos},
  author = {Le, Cuong and Johannson, Viktor and Kok, Manon and Wandt, Bastian},
  journal = {Arxiv},
  year = {2024}
}
```

## Acknownledgement
