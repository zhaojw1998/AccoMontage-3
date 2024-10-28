# Structured Accompaniment Arrangement
[![arXiv](https://img.shields.io/badge/arXiv-2310.16334-brightgreen.svg?logo=arXiv&style=flat-round)](https://arxiv.org/abs/2310.16334)
[![GitHub](https://img.shields.io/badge/GitHub-demo%20page-blue?logo=Github&style=flat-round)](https://zhaojw1998.github.io/structured-arrangement/)
[![Colab](https://img.shields.io/badge/Colab-tutorial-blue?logo=googlecolab&style=flat-round)](https://colab.research.google.com/drive/1LSY1TTkSesDUfpJplq5xi-3-DI09fWQ9?usp=sharing)

We present a two-stage ststem for *whole-song*, *multi-track* accompaniment arrangement. Initially, a piano accompaniment is generated given a lead sheet; subsequently, a multi-track arrangement is orchestrated with control on choices of instruments. Our main novelty lies in the second stage, where we implement long-term *style prior modelling* based on disentangled music content/style factors. Please refer to our [paper](https://arxiv.org/abs/2310.16334) for the detailed work.

Demp page: https://zhaojw1998.github.io/structured-arrangement/

Try our system [on Colab](https://colab.research.google.com/drive/1LSY1TTkSesDUfpJplq5xi-3-DI09fWQ9?usp=sharing)

### Code and File Directory
This repository is organized as follows:
```
root
  ├──data_processing/                   scripts for data processing
  │    
  ├──demo/                              MIDI pieces for demonstration
  │       
  ├──orchestrator/                      the orchestrator module at Stage 2
  │    
  ├──piano_arranger/                    the piano arranger module at Stage 1
  │    
  ├──test/                              scripts and results for objective evaluation
  │   
  ├──arrangement_utils.py               functionals for model inference
  │   
  ├──inference_arrangement.ipynb        two-stage model inference (lead sheet via piano to multi-track)
  │ 
  ├──inference_orchestration.ipynb      Stage-2 module inference (piano to multi-track)
  │ 
  ├──train_autoencoder.py               training script for Stage-2 autoencoder
  │ 
  └──train_prior.py                     training script for Stage-2 prior model
```

### How to run
* Try out our system on [Google Colab](https://colab.research.google.com/drive/1LSY1TTkSesDUfpJplq5xi-3-DI09fWQ9?usp=sharing), where you can quickly test it online.

* Alternatively, follow the guidelines in [`inference_arrangement.ipynb`](./inference_arrangement.ipynb) offline for more in-depth testing. 

* To train our model from scratch, first [`train_autoencoder.py`](./train_autoencoder.py), and then [`train_prior.py`](./train_prior.py). You may wish to configure a few params such as `BATCH_SIZE` from the beginning of the training scripts. When `DEBUG_MODE`=1, it will load a small portion of data and quickly run through for debugging purpose. 

To set up the environment, run the following:
```
# python version <= 3.9 is recommended
conda create -n env python=3.8
conda activate env
# install pytorch
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# install other dependencies
pip install -r requirements.txt
```

### Data and Checkpoints

* Model checkpoints can be downloaded [via this link](https://drive.google.com/file/d/1ZyswS0p_t2Ij5vyaFkM5IbVgphf78oTB/view?usp=sharing).

* Processed dataset (LMD) for training the prior model can be downloaded [via this link](https://drive.google.com/file/d/14BHxnYDYSuGe0m3XXqIPL1-d4376GOBH/view?usp=sharing).

* Processed dataset (Slakh2100) for training the autoencoder is accessbible [in this repo](https://github.com/zhaojw1998/Query-and-reArrange/tree/main/data/Slakh2100).


### Contact
Jingwei Zhao (PhD student in Data Science at NUS)

jzhao@u.nus.edu

Oct. 28, 2024