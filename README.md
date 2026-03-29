# FDE-Net: A Frequency-Domain Enhanced Network for Ship Detection in Remote Sensing Images

Welcome to use the code from our paper "FDE-Net: A Frequency-Domain Enhanced Network for Ship Detection in Remote Sensing Images".

## Environment
- Python 3.9+
- PyTorch >= 2.7.0

## Dataset
Levir-Ship, AI-TOD, and HRSC2016.

## Training and Testing
```bash
yolo detect train model=FDENet.yaml data=... batch=... epochs=...
```

## Requirements
We use a single RTX 3090 24G GPU for training and evaluation.

## Note
Currently, we provide only preprocessing and training/testing scripts. The full model code will be released upon paper acceptance.

## Contact
If you have any questions, don't hesitate to contact me via yena@stu.cqut.edu.cn.