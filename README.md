# FedCSN: Mitigating Skewed Feature via Class-wise Selective Normalization in Federated Learning

Official PyTorch implementation of **FedCSN**, a class-wise selective normalization framework for federated learning under feature distribution skew.

## Overview

FedCSN addresses the limitation of existing normalization-based augmentation methods in federated learning, where indiscriminate cross-class statistic sharing may introduce invalid augmentation noise under heterogeneous feature distributions.

The proposed method preserves semantic consistency during normalization augmentation through:

- **Class-wise statistic sharing**
- **Dynamic dominant-class statistic selection**
- **Fallback mechanism for missing classes**
- **Max-K guardrail mechanism**

Extensive experiments on Office-Caltech-10, Office-Home, and DomainNet demonstrate the effectiveness of FedCSN under severe Non-IID settings.

---

## Environment

- Python: 3.10+
- PyTorch: 2.8.0

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```bash
FedCSN/
│
├── main.py
├── methods.py
├── utils.py
├── nets.py
├── requirements.txt
│
├── data/
│   ├── office_caltech_10/
│   ├── office_home/
│   └── domainnet/
│
└── README.md
```

---

## Datasets

### Office-Caltech-10

Place the processed `.pkl` files under:

```bash
data/office_caltech_10/
```

Expected files:

```bash
amazon_train.pkl
amazon_test.pkl
caltech_train.pkl
caltech_test.pkl
dslr_train.pkl
dslr_test.pkl
webcam_train.pkl
webcam_test.pkl
```

---

### DomainNet

Place the processed `.pkl` files under:

```bash
data/domainnet/
```

Expected files:

```bash
clipart_train.pkl
clipart_test.pkl
infograph_train.pkl
infograph_test.pkl
painting_train.pkl
painting_test.pkl
quickdraw_train.pkl
quickdraw_test.pkl
real_train.pkl
real_test.pkl
sketch_train.pkl
sketch_test.pkl
```

---

## Running Experiments

Run training with:

```bash
python main.py
```

Experiment configurations and hyperparameters are defined in the argument settings inside the code files and can be modified manually as needed.

---

## Core Components

FedCSN mainly consists of:

- **Class-wise Normalization Augmentation**
- **Dynamic Statistic Selection**
- **Fallback Mechanism**
- **Max-K Guardrail Mechanism**

The implementation of these components can be found in:

- `methods.py`
- `utils.py`

---

## Notes

- The repository currently provides the processed `.pkl` files for Office-Caltech-10 and DomainNet experiments.
