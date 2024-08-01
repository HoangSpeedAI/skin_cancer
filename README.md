# Skin Cancer Classification

This project aims to classify skin cancer images using deep learning techniques, specifically leveraging PyTorch and transfer learning with ResNet.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Validation](#validation)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/skin-cancer-classification.git
   cd skin-cancer-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n skin_cancer python=3.9
   conda activate skin_cancer
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Download ZIP file from [here](https://urldefense.proofpoint.com/v2/url?u=https-3A__d4h9zj04.na1.hs-2Dsales-2Dengage.com_Ctc_L1-2B23284_d4H9ZJ04_JlF2-2D6qcW8wLKSR6lZ3nMW5pc0-5FL1QRPYXW6KpVNP1xxYw6W8Mx97p3Y66h-5FW6MPwxl2CyrnGW8J8CgN69SWhWW2fKTZ52-5F6K6tW1HDzsL2CWPrKW7DYzvV47jv81W3-5FQLJW5CMx2FW7sBTzw8H94pGW70tjvq6Np0mBW3-2DG1lz4kKy8cW8Z2gr28r0F3FW7DHDtW2t6DfYW2JPzl93KkVq9N4NzSXhtvCNHW6vmpp25SpBQ8W5JM-5FRj4CvYKPVv4ntL4kc78MW8hSMg45grwnzW8bR6hx2D8-2DKGW6SLMZL8dqyfsW8NctCP54NGGcW8d1LZX5LNZ51VxQh575VQYccW35nBKZ7csGx5VvHHRB5X8s5PW8gLT2g76bqwjf4V5hFj04&d=DwMFaQ&c=0eTCB0nNi4K2jeP4BAvnzcTY3bBUzv8iW-Zx9Sm4vP0&r=X8ZeqHd4hWrdn2JHTyGJVFVcupB1rkbgUcMu_GxklcM&m=w8Laz3dSz9tquwo9jlXrmWZsTu0yrUCKfkHVif-si764egsnrXPn2Zz-xaPqCbKH&s=PQT-XPUsJAVCM-LDGST5A3jaXgCLNR0OFqoFzWl3R28&e=)


Prepare your dataset in the `data/` directory using the following structure:
```
data/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    └── val/
        ├── class1/
        ├── class2/
```

## Training the Model

To train the model, run:

```bash
python src/main.py --config config.yaml
```

### Configuration

Edit the `config.yaml` file to customize parameters such as batch size, learning rate, and number of epochs.

## Validation

The model's performance is evaluated on a validation set after each training epoch. You can monitor validation loss and accuracy in the console output.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.