# Temporal Relational Modeling with Self-Supervision for Action Segmentation (AAAI'21)
This repository provides a PyTorch implementation of the paper [Temporal Relational Modeling with Self-Supervision for Action Segmentation](https://arxiv.org/abs/2012.07508).

![](pipeline.png =100x20)

Tested with:
- PyTorch 1.6.0
- Python 3.6.12

### Training:

* Download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels. (~30GB). We use the same data with [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://github.com/yabufarha/ms-tcn)
* Extract it so that you have the `data` folder in the same directory as `main.py`.
* To train the model run `python main.py --action=train --dataset=DS --split=SP --num_stages=4 --num_layers=10 --num_f_maps=64 --df_size=3` where `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other datasets. `num_stages`, `num_layers` and `df_size` are the hyper-parameters of the proposed model, which can be changed to other values.

### Prediction:

Run `python main.py --action=predict --dataset=DS --split=SP --num_stages=4 --num_layers=10 --num_f_maps=64 --df_size=3`. 

### Evaluation:

Run `python eval.py --dataset=DS --split=SP --num_stages=4 --num_layers=10 --num_f_maps=64 --df_size=3`. 

### Citation:

If you use the code, please cite

    D. Wang, H. Di, X. Li, and D. Dou.
    Temporal Relational Modeling with Self-Supervision for Action Segmentation.
    In AAAI Conference on Artificial Intelligence(AAAI), 2021

### Acknowlegements
This code is borrowed or adapted from [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://github.com/yabufarha/ms-tcn). Thanks a lot for their great work!