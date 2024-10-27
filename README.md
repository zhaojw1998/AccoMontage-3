# Structured Accompaniment Arrangement
[![arXiv](https://img.shields.io/badge/arXiv-2310.16334-brightgreen.svg?logo=arXiv&style=flat-round)](https://arxiv.org/abs/2310.16334)
[![GitHub](https://img.shields.io/badge/GitHub-demo%20page-blue?logo=Github&style=flat-round)](https://zhaojw1998.github.io/structured-arrangement/)
[![Colab](https://img.shields.io/badge/Colab-tutorial-blue?logo=googlecolab&style=flat-round)](https://colab.research.google.com/drive/1LSY1TTkSesDUfpJplq5xi-3-DI09fWQ9?usp=sharing)

Repository for Paper: [Zhao et al., Structured Multi-Track Accompaniment Arrangement via Style Prior Modelling, in NeurIPS 2024](https://arxiv.org/abs/2310.16334).

We present a two-stage ststem for *whole-song*, *multi-track* accompaniment arrangement. In the first stage, a piano accompaniment is generated given a lead sheet. In the second stage, a multi-track accompaniment is orchestrated with customizable track numbers and choices of instruments. Our main novelty (essentials of this repo) lies in the second stage, where we implement long-term *style prior modelling* based on disentangled music content and style factors. Please refer to our paper for the detailed work.

Demp page: https://zhaojw1998.github.io/structured-arrangement/

Our system can be quickly tested [on Colab](https://colab.research.google.com/drive/1LSY1TTkSesDUfpJplq5xi-3-DI09fWQ9?usp=sharing).


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
  ├──inference_arrangement.ipynb        two-stage model inference (arrangement from lead sheet)
  │ 
  ├──inference_orchestration.ipynb      Stage-2 module inference (orchestration from piano)
  │ 
  ├──train_autoencoder.py               training script for Stage-2 autoencoder
  │ 
  └──train_prior.py                     training script for Stage-2 prior model
```


### How to run
* You can quckly test our system on [Google Colab](https://colab.research.google.com/drive/1LSY1TTkSesDUfpJplq5xi-3-DI09fWQ9?usp=sharing), where you can quickly test our model online.

* Alternatively, follow the guidelines in [`./inference_arrangement.ipynb`](./inference_arrangement.ipynb) offline for more in-depth testing. 

* If you wish to train our model from scratch, run [`./train_prior.py`](./train_prior.py). Please first download our processed LMD dataset and configure the corresponding data directory in the script. You may also wish to configure a few params such as `BATCH_SIZE` from the beginning of the script. When `DEBUG_MODE`=1, it will load a small portion of data and quickly run through for debugging purpose.


### Data and Checkpoints

* Model checkpoints can be downloaded [via this link](https://drive.google.com/file/d/1ZyswS0p_t2Ij5vyaFkM5IbVgphf78oTB/view?usp=sharing).

* Processed dataset (LMD) for training the prior model can be downloaded [via this link](https://drive.google.com/file/d/14BHxnYDYSuGe0m3XXqIPL1-d4376GOBH/view?usp=sharing).

* Processed dataset (Slakh2100) for training the autoencoder is accessbible [in this repo](https://github.com/zhaojw1998/Query-and-reArrange/tree/main/data/Slakh2100).


### Contact
Jingwei Zhao (PhD student in Data Science at NUS)

jzhao@u.nus.edu

Oct. 27, 2024