# SSFL-IDS

This repository contains the code for the paper:
<br>
***Semi-Supervised Federated Learning Based Intrusion Detection Method for Internet of Things***
<br>
In IEEE Internet of Things Journal, doi: 10.1109/JIOT.2022.3175918.

## Overview of SSFL-IDS

<img src="https://github.com/iZRJ/SSFL-IDS/blob/main/figure/SSFL-IDS.png">

Overview of proposed semi-supervised federated learning scheme for intrusion detection. Our CNN-based classifier is trained on the labeled local data with supervised training and on unlabeled open data with distillation training. Furthermore, we introduce multiple mechanisms to jointly improve the quality of global labels.

## Schematic Illustration of Client Data 

<img src="https://github.com/iZRJ/SSFL-IDS/blob/main/figure/Scenario-non-IID.png" width="500">

The schematic illustration of samples per class allocated to each client, where the x-axis indicates client IDs, the y-axis indicates class labels, and the size of scattered points indicates the number of labeled samples.

## Dependency

```
torch=1.9.0
numpy=1.19.5
scikit-learn=0.24.2
```

## Code

### 1. Unzip Dataset

```
cd data
unzip nba_iot_1000.zip
```

### 2. Train SSFL-IDS

```
python SSFL-IDS.py
```

## Contact-Info
[Ruijie Zhao](https://github.com/iZRJ)
<br>
Email: ruijiezhao@sjtu.edu.cn

## About
Link to our laboratory: [SJTU-NSSL](https://github.com/NSSL-SJTU "SJTU-NSSL")

## Reference

R. Zhao, Y. Wang, Z. Xue, T. Ohtsuki, B. Adebisi, and G. Gui, ``Semi-Supervised Federated Learning Based Intrusion Detection Method for Internet of Things,'' IEEE Internet Things J., early access, doi: 10.1109/JIOT.2022.3175918.

```
@ARTICLE{SSFL_IDS,
  author    = {Zhao, Ruijie and Wang, Yijun and Xue, Zhi and Ohtsuki, Tomoaki and Adebisi, Bamidele and Gui, Guan},
  title     = {Semi-Supervised Federated Learning Based Intrusion Detection Method for Internet of Things},
  booktitle = {IEEE Internet of Things Journal},
  pages     = {1--14},
  doi={10.1109/JIOT.2022.3175918}},
  year      = {2022}}
```