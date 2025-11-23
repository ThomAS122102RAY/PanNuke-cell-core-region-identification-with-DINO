# PanNuke Nuclei Segmentation with DINO Self-Supervision

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

This project implements a deep learning pipeline for nuclei segmentation and classification on the [PanNuke dataset](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification) (Kaggle Mirror, split into 3 parts). It leverages **DINO (Self-distillation with no labels)** to pre-train a Vision Transformer (ViT) backbone on histopathology images, followed by a segmentation task using a TransUNet-inspired architecture.

## 🌟 Key Features

*   **End-to-End Pipeline**: Automated workflow from data preprocessing to inference.
*   **Self-Supervised Learning**: Uses DINO to learn robust feature representations from unlabeled histopathology patches.
*   **Advanced Segmentation**: Fine-tunes a segmentation model (TransUNet) using the pre-trained ViT backbone.
*   **Efficient Data Handling**: Uses memory mapping (`numpy.memmap`) to handle large datasets without overloading RAM.
*   **Visualization**: Built-in tools to generate overlay visualizations of predictions vs. ground truth.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ThomAS122102RAY/PanNuke-cell-core-region-identification-with-DINO.git
    cd PanNuke-cell-core-region-identification-with-DINO
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    **Requirements:**
    *   Python 3.8+
    *   PyTorch >= 1.9.0
    *   torchvision
    *   timm
    *   albumentations
    *   opencv-python
    *   scikit-learn
    *   matplotlib

## 📂 Dataset Preparation

1.  Download the **PanNuke Dataset** from Kaggle (see [DOWNLOAD_INSTRUCTIONS.md](DOWNLOAD_INSTRUCTIONS.md) for links to all 3 parts).
2.  Extract and organize the data in the `data/pannuke` directory as follows:
    ```
    data/pannuke/
    ├── Fold 1/
    │   ├── 1_images.npy  (or images.npy)
    │   ├── 1_masks.npy   (or masks.npy)
    │   └── 1_types.npy   (or types.npy)
    ├── Fold 2/
    │   ├── 2_images.npy
    │   ├── 2_masks.npy
    │   └── 2_types.npy
    └── Fold 3/
        ├── 3_images.npy
        ├── 3_masks.npy
        └── 3_types.npy
    ```
    **Note**: The pipeline supports both naming conventions (`images.npy` or `1_images.npy`).
    
3.  **Automatic Split**: When you run the main pipeline, it will automatically:
    *   Copy files from `Fold 1` and `Fold 2` → `data/pannuke/train/`
    *   Copy files from `Fold 3` → `data/pannuke/test/`
    *   Merge multiple `.npy` shards into single files for efficient training


## 🚀 Usage

The project is controlled by a single main script that handles the entire workflow.

### Run the Pipeline

```bash
python main_pipeline_fixed_dino_guard.py
```

You will be prompted to select a mode:

1.  **Train DINO (train)**:
    *   Extracts patches from the dataset.
    *   Runs DINO self-supervised pre-training.
    *   Trains the segmentation head.
    *   Evaluates and visualizes results.

2.  **Finetune with Official Weights (finetune)**:
    *   Downloads/Uses official DINO ViT-Small weights.
    *   Fine-tunes the segmentation model on your data.

3.  **Inference (inference)**:
    *   Uses an existing checkpoint to generate predictions and visualizations.

### Visualization Only

If you want to generate visualizations for existing predictions:

```bash
python visualize_predictions.py --split test --num_samples 20
```

## 📁 Project Structure

```
.
├── main_pipeline_fixed_dino_guard.py  # Main entry point
├── visualize_predictions.py           # Visualization utility
├── requirements.txt                   # Dependencies
├── data/                              # Dataset directory
├── dino/                              # DINO algorithm implementation
├── segmentor/                         # Segmentation model & training
│   ├── train_segmentor.py
│   ├── predict_segmentor.py
│   └── transunet.py                   # Model architecture
├── preprocess/                        # Data loading & processing
└── checkpoints/                       # Saved models
```

## 🚀 Standalone Inference

A standalone inference tool is available in `pannuke_inference_dist/` for easy prediction on new images.

### Quick Start (Interactive Mode)

Simply run the script directly - it will prompt you for the model path:
```bash
cd pannuke_inference_dist
python inference.py
```
You'll be asked to provide:
- Input folder path (defaults to `data/pannuke/test` if available)
- Model weights path (`.pth` file)

### Command Line Mode

For batch processing or automation:

**For regular images (jpg/png)**:
```bash
python inference.py \
    --input "C:/path/to/images" \
    --output "results" \
    --weights "path/to/model_final.pth"
```

**For PanNuke format (.npy files)**:
```bash
python inference.py \
    --input "C:/path/to/pannuke/folder" \
    --output "results" \
    --weights "path/to/model_final.pth" \
    --pannuke_format
```

### Output Files

The inference generates three types of visualizations for each image:
- `*_mask.png` - Binary segmentation mask (white=cells, black=background)
- `*_overlay.jpg` - **Dual-color overlay** (Blue=Cells, Yellow=Background) blended with original image
- `*_combined.jpg` - Side-by-side comparison (original | mask)

**Pre-trained Model**: Download from [Google Drive](https://drive.google.com/file/d/1rpBq-JBCZovthO9diK5G1jJEAiNiZTON/view?usp=drive_link)

## 📊 Results

<img width="426" height="177" alt="螢幕擷取畫面 2025-11-23 005802" src="https://github.com/user-attachments/assets/e31e5486-296f-4134-9a69-b1bd3f0a81e5" />

The pipeline generates:

*   **Metrics**: IoU and Dice scores for Background and Cells.
*   **Visualizations**: Saved in `visualizations/` and `inference_results/` folders, showing:
    *   Original Image
    *   Ground Truth Mask (training only)
    *   Predicted Mask
    *   **Dual-Color Overlay**: Blue for Cells, Yellow for Background
    *   Side-by-side comparisons for easy evaluation

<img width="1771" height="1194" alt="sample_0456" src="https://github.com/user-attachments/assets/5ce438a8-f084-49ef-9509-bd2af8be87c9" />
<img width="1771" height="1194" alt="sample_0419" src="https://github.com/user-attachments/assets/3cc920cc-f6ec-4c36-a183-7479b302b458" />
<img width="1771" height="1194" alt="sample_0356" src="https://github.com/user-attachments/assets/b6611a1c-06cc-4ced-8df6-1fc830c32565" />
<img width="1771" height="1194" alt="sample_0102" src="https://github.com/user-attachments/assets/c4fb82fb-1841-44e7-8890-fc275b0e1a6f" />
<img width="1771" height="1194" alt="sample_2619" src="https://github.com/user-attachments/assets/ac07348a-bd32-4a83-aff1-1ea22a98aecf" />
<img width="1771" height="1194" alt="sample_2233" src="https://github.com/user-attachments/assets/cf9102a0-bd11-41a5-aaa7-ad69932b79b7" />
<img width="1771" height="1194" alt="sample_1126" src="https://github.com/user-attachments/assets/5502ce7c-ced5-470c-9ded-61dc55b4f728" />
<img width="1771" height="1194" alt="sample_1003" src="https://github.com/user-attachments/assets/4ebaac67-3c06-4f77-84b4-7e2b44ea8a53" />
<img width="1771" height="1194" alt="sample_0914" src="https://github.com/user-attachments/assets/489e2eaf-fb6d-4be0-b533-2ec8e368efb1" />
<img width="1771" height="1194" alt="sample_0571" src="https://github.com/user-attachments/assets/6ee45fa0-c886-4144-a2d0-73195e52c489" />
