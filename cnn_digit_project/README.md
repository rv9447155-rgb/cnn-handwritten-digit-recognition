# Reproducing Gradient-Based Learning with a Small CNN

A student-scale reproduction inspired by:

> Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-Based Learning Applied to Document Recognition,” *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998. DOI: 10.1109/5.726791.

## Project idea
The paper explains why convolutional neural networks are well suited to handwritten-character recognition. This project implements a compact CNN on scikit-learn's 8×8 handwritten-digits dataset and evaluates its classification performance.

This is **inspired by the paper, not an exact reproduction**: the original work used larger document-recognition systems and datasets, while this project uses a small public dataset for reproducibility on a normal laptop.

## Dataset
The project uses `sklearn.datasets.load_digits`: 1,797 grayscale images, 10 classes, each represented as an 8×8 image with pixel values 0–16.

## Model
- Conv2D(1→16, 3×3) + ReLU + MaxPool
- Conv2D(16→32, 3×3) + ReLU + MaxPool
- Flatten
- Dense 64 + ReLU + Dropout
- Dense 10 + softmax for prediction

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

Training creates `results/metrics.json`, `results/training_accuracy.png`, `results/confusion_matrix.png`, and the model weights.

## Repository structure
```text
cnn-digit-research-project/
├── src/
│   ├── train.py
│   └── predict.py
├── results/
│   ├── metrics.json
│   ├── history.json
│   ├── training_accuracy.png
│   ├── confusion_matrix.png
│   └── cnn_digit_model.pt
├── docs/
├── presentation/
├── requirements.txt
├── README.md
└── .gitignore
```

## Academic honesty
Do not present the student implementation as the original authors' implementation. State clearly that it is a small educational adaptation inspired by the paper.

## References
1. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324. DOI: 10.1109/5.726791.
2. scikit-learn documentation, `load_digits` dataset.
