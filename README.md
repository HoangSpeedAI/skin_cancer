# Skin Cancer Classification

This project aims to classify skin cancer images using deep learning techniques.

<!-- ![Sample Image](viz_train.png) -->

## Table of Contents

- [Technique Highlights](#technique-highlights)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Statistics](#dataset-statistics)
- [Configuration](#configuration)
- [Training](#training)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [License](#license)

## Technique Highlights

- **Data Split**: Training and validation data split in an 80/20 ratio.
- **Data Augmentation**: Applied techniques to enhance model robustness.
- **Regularization**: Used L2 regularization (weight decay) with the Adam optimizer.
- **Transfer Learning**: Leveraged pretrained architectures (ResNet and EfficientNet).
- **Learning Rate Schedulers**: Integrated StepLR, CycleLR, and CosineLR.
- **Gradient Clipping**: Stabilized training to prevent exploding gradients.
- **Weight Freezing**: Did not freeze pretrained weights, allowing full adaptation.
- **Model Checkpointing**: Saved the best model at the end of each epoch.

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

Download the [dataset](https://urldefense.proofpoint.com/v2/url?u=https-3A__d4h9zj04.na1.hs-2Dsales-2Dengage.com_Ctc_L1-2B23284_d4H9ZJ04_JlF2-2D6qcW8wLKSR6lZ3nMW5pc0-5FL1QRPYXW6KpVNP1xxYw6W8Mx97p3Y66h-5FW6MPwxl2CyrnGW8J8CgN69SWhWW2fKTZ52-5F6K6tW1HDzsL2CWPrKW7DYzvV47jv81W3-5FQLJW5CMx2FW7sBTzw8H94pGW70tjvq6Np0mBW3-2DG1lz4kKy8cW8Z2gr28r0F3FW7DHDtW2t6DfYW2JPzl93KkVq9N4NzSXhtvCNHW6vmpp25SpBQ8W5JM-5FRj4CvYKPVv4ntL4kc78MW8hSMg45grwnzW8bR6hx2D8-2DKGW6SLMZL8dqyfsW8NctCP54NGGcW8d1LZX5LNZ51VxQh575VQYccW35nBKZ7csGx5VvHHRB5X8s5PW8gLT2g76bqwjf4V5hFj04&d=DwMFaQ&c=0eTCB0nNi4K2jeP4BAvnzcTY3bBUzv8iW-Zx9Sm4vP0&r=X8ZeqHd4hWrdn2JHTyGJVFVcupB1rkbgUcMu_GxklcM&m=w8Laz3dSz9tquwo9jlXrmWZsTu0yrUCKfkHVif-si764egsnrXPn2Zz-xaPqCbKH&s=PQT-XPUsJAVCM-LDGST5A3jaXgCLNR0OFqoFzWl3R28&e=) and unzip it to the `data` directory. The structure should look like this:
```
data/
    ├── train/
    │   ├── Benign/
    │   ├── Malignant/
    └── test/
        ├── Benign/
        ├── Malignant/
```

## Dataset Statistics

| Type      | Benign | Malignant |
|-----------|--------|-----------|
| Training  | 5.031  | 4,472     |
| Validation| 1,258  | 1,118     |
| Testing   | 1,000  | 1,000     |

### Configuration

Edit the `config.yaml` file to customize parameters such as batch size, learning rate, and number of epochs.
```
batch_size: 32
num_epochs: 20
learning_rate: 0.001
data_path: "data"

input_size: 224
input_channels: 3
output_channels: 1

architecture: "resnet50"
scheduler: 'cyclic'
step_size_up: 10
```

## Training

To train the model, run:

```bash
python src/main.py -c configs/efficientnet_b3_config.yaml -m train
```

Here is the output from training the EfficientNet-B3 model:
```
Epoch 1/20: 100%|██████████████████████████████████████████| 297/297 [01:36<00:00,  3.07batch/s, loss=0.61, lr=1e-5]
Epoch [1/20], Train Loss: 0.6104, Validation Loss: 0.4904, Accuracy: 0.8585
--> Saved best model to experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep1_0.8585.pth
Epoch 2/20: 100%|██████████████████████████████████████| 297/297 [01:34<00:00,  3.16batch/s, loss=0.29, lr=0.000109]
Epoch [2/20], Train Loss: 0.2904, Validation Loss: 0.3197, Accuracy: 0.8745
--> Removed old best model at experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep1_0.8585.pth
--> Saved best model to experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep2_0.8745.pth
Epoch 3/20: 100%|██████████████████████████████████████| 297/297 [01:33<00:00,  3.16batch/s, loss=0.26, lr=0.000208]
Epoch [3/20], Train Loss: 0.2598, Validation Loss: 0.2859, Accuracy: 0.8901
--> Removed old best model at experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep2_0.8745.pth
--> Saved best model to experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep3_0.8901.pth
Epoch 4/20: 100%|█████████████████████████████████████| 297/297 [01:34<00:00,  3.15batch/s, loss=0.258, lr=0.000307]
Epoch [4/20], Train Loss: 0.2580, Validation Loss: 0.2513, Accuracy: 0.9086
--> Removed old best model at experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep3_0.8901.pth
--> Saved best model to experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep4_0.9086.pth
Epoch 5/20: 100%|█████████████████████████████████████| 297/297 [01:34<00:00,  3.13batch/s, loss=0.232, lr=0.000406]
Epoch [5/20], Train Loss: 0.2322, Validation Loss: 0.2564, Accuracy: 0.9006
Epoch 6/20: 100%|██████████████████████████████████████| 297/297 [01:34<00:00,  3.13batch/s, loss=0.23, lr=0.000505]
Epoch [6/20], Train Loss: 0.2300, Validation Loss: 0.2518, Accuracy: 0.9006
Epoch 7/20: 100%|█████████████████████████████████████| 297/297 [01:33<00:00,  3.17batch/s, loss=0.239, lr=0.000604]
Epoch [7/20], Train Loss: 0.2391, Validation Loss: 0.2819, Accuracy: 0.9044
Epoch 8/20: 100%|█████████████████████████████████████| 297/297 [01:33<00:00,  3.17batch/s, loss=0.239, lr=0.000703]
Epoch [8/20], Train Loss: 0.2389, Validation Loss: 0.2461, Accuracy: 0.9061
--> Removed old best model at experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep4_0.9086.pth
--> Saved best model to experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep8_0.9061.pth
Epoch 9/20: 100%|██████████████████████████████████████| 297/297 [01:33<00:00,  3.16batch/s, loss=0.23, lr=0.000802]
Epoch [9/20], Train Loss: 0.2300, Validation Loss: 0.3493, Accuracy: 0.8800
Epoch 10/20: 100%|█████████████████████████████████████| 297/297 [01:33<00:00,  3.18batch/s, loss=0.24, lr=0.000901]
Epoch [10/20], Train Loss: 0.2400, Validation Loss: 0.2529, Accuracy: 0.8914
Epoch 11/20: 100%|███████████████████████████████████████| 297/297 [01:33<00:00,  3.16batch/s, loss=0.239, lr=0.001]
Epoch [11/20], Train Loss: 0.2388, Validation Loss: 0.2451, Accuracy: 0.9027
--> Removed old best model at experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep8_0.9061.pth
--> Saved best model to experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep11_0.9027.pth
Epoch 12/20: 100%|████████████████████████████████████| 297/297 [01:36<00:00,  3.09batch/s, loss=0.233, lr=0.000901]
Epoch [12/20], Train Loss: 0.2333, Validation Loss: 0.2349, Accuracy: 0.9057
--> Removed old best model at experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep11_0.9027.pth
--> Saved best model to experiments/efficientnet_b3_20240803_071931/efficientnet_b3_ep12_0.9057.pth
Epoch 13/20: 100%|████████████████████████████████████| 297/297 [01:34<00:00,  3.15batch/s, loss=0.208, lr=0.000802]
Epoch [13/20], Train Loss: 0.2084, Validation Loss: 0.2444, Accuracy: 0.8981
Epoch 14/20: 100%|████████████████████████████████████| 297/297 [01:33<00:00,  3.16batch/s, loss=0.199, lr=0.000703]
Epoch [14/20], Train Loss: 0.1988, Validation Loss: 0.2707, Accuracy: 0.8943
Epoch 15/20: 100%|████████████████████████████████████| 297/297 [01:35<00:00,  3.10batch/s, loss=0.182, lr=0.000604]
Epoch [15/20], Train Loss: 0.1816, Validation Loss: 0.2611, Accuracy: 0.8914
Epoch 16/20: 100%|████████████████████████████████████| 297/297 [01:35<00:00,  3.11batch/s, loss=0.175, lr=0.000505]
Epoch [16/20], Train Loss: 0.1747, Validation Loss: 0.2569, Accuracy: 0.9027
Epoch 17/20: 100%|████████████████████████████████████| 297/297 [01:34<00:00,  3.16batch/s, loss=0.163, lr=0.000406]
Epoch [17/20], Train Loss: 0.1627, Validation Loss: 0.2463, Accuracy: 0.9095
Epoch 18/20: 100%|████████████████████████████████████| 297/297 [01:35<00:00,  3.10batch/s, loss=0.138, lr=0.000307]
Epoch [18/20], Train Loss: 0.1377, Validation Loss: 0.2980, Accuracy: 0.9044
Epoch 19/20: 100%|████████████████████████████████████| 297/297 [01:35<00:00,  3.11batch/s, loss=0.125, lr=0.000208]
Epoch [19/20], Train Loss: 0.1248, Validation Loss: 0.2797, Accuracy: 0.9091
Epoch 20/20: 100%|████████████████████████████████████| 297/297 [01:35<00:00,  3.10batch/s, loss=0.101, lr=0.000109]
Epoch [20/20], Train Loss: 0.1014, Validation Loss: 0.2968, Accuracy: 0.9040
```

## Testing

To test the model, run:

```bash
python src/main.py -c configs/efficientnet_b1_config.yaml -m test -mp experiments/efficientnet_b1_20240803_111717/efficientnet_b1_ep4_0.9149.pth
```

Here is the output from training the EfficientNet-B3 model:
```
Accuracy: 0.9187
Precision: 0.9403
Recall: 0.9140
F1 Score: 0.9270
```

## Evaluation

| Model                | Accuracy  | Precision  | Recall    | F1 Score  |
|----------------------|-----------|------------|-----------|-----------|
| Resnet18             | 0.8120    | **0.9561** | 0.8050    | 0.8740    |
| Resnet50             | *0.8680*| 0.9420     | *0.8610*| *0.8997*|
| EfficientNet-B1     | 0.8293    | 0.9570     | 0.8230    | 0.8849    |
| **EfficientNet-B3**  | **0.9187**| 0.9403     | **0.9140**| **0.9270**|

- **Best**: Highlighted in bold.
- *Second Best*: Underlined.

## Conclusion

The evaluation metrics indicate the following:

- **EfficientNet-B3** is the top performer across all metrics, showcasing the highest accuracy, recall, F1 score, and demonstrating a strong balance in performance.
- **Resnet50** is the second-best model, particularly excelling in accuracy and recall.
- **Resnet18** exhibits the highest precision, making it reliable for positive predictions, though it ranks lower in other metrics.
- Overall, **EfficientNet-B3** stands out as the most effective model, while **Resnet50** serves as a strong alternative, especially in terms of accuracy and recall.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.