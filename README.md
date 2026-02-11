# Brazilian Document Classification

Train a document category model to classify Brazilian document types (e.g. CNH, CPF, RG, etc.) using 60% of the dataset for training and 40% for testing.

## Setup

```bash
cd brazillian_document_recongnition
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Dataset Structure

Place your raw dataset in:

```
data/raw/
  CNH/        # National Driver's License
  CPF/        # Tax ID document
  RG/         # General Registration
  ...         # other document types (8–9 total)
```

Each folder should contain images (`.jpg`, `.png`, etc.) of that document type.

## Usage

### 1. Split dataset (60% train / 40% test)

```bash
python split_data.py
```

### 2. Train model

```bash
python train.py
```

### 3. Evaluate on held-out test set

```bash
python evaluate.py
```

## Project Structure

```
brazillian_document_recongnition/
├── config.py          # Configuration
├── split_data.py      # 60/40 split
├── train.py           # Training (60% of data)
├── evaluate.py        # Test accuracy (40% held-out)
├── src/
│   ├── dataset.py     # Dataset & transforms
│   └── model.py       # ResNet18 classifier
├── data/              # Created by split_data.py
│   ├── raw/           # Place your dataset here
│   ├── train/         # 60% training
│   └── test/          # 40% testing
├── checkpoints/       # Saved models
└── requirements.txt
```

## License

MIT
