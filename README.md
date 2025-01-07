# Wastage Classification using Vision Transformers

## Project Overview
This project aims to classify images of waste types using a Vision Transformer (ViT) model. The notebook leverages the Hugging Face `transformers` and `datasets` libraries for model training and evaluation, focusing on garbage-type image detection using the pretrained model `dima806/garbage_types_image_detection`.

Additionally, the project includes an integrated chatbot that provides suggestions for recyclable projects based on the classified waste type. This feature enhances the usability of the system by offering actionable recommendations for waste recycling.

## Features
- **Preprocessing**: Image augmentation and normalization for improved model performance.
- **Model Training**: Fine-tuning the Vision Transformer for image classification.
- **Evaluation**: Detailed metrics including accuracy, F1-score, confusion matrix, and more.
- **Robustness**: Handles corrupted image files gracefully.
- **Chatbot Integration**: Provides project ideas for recycling based on waste classification results.

---

## Installation

To replicate the environment, follow these steps:


## Usage

### Dataset
Download the dataset from [dataset link](#) and place it in the `data/` directory. Ensure the folder structure matches the expected format:
```
data/
    train/
    val/
    test/
```

### Running the Notebook
Open the Jupyter Notebook and run all cells sequentially:
```bash
jupyter notebook wastage-classification.ipynb
```

### Using the Chatbot
After classification, the chatbot can be run to suggest recycling project ideas. This feature provides personalized recommendations to promote waste reuse and recycling.

---

## Code Structure

### Preprocessing
- Uses `torchvision` transformations for resizing, normalization, and data augmentation.
- Applies separate transformations for training and validation datasets.

### Model
- Fine-tunes the pretrained Vision Transformer `dima806/garbage_types_image_detection`.
- Employs `Hugging Face` `Trainer` API for streamlined training and evaluation.

### Metrics
- Computes accuracy, F1-score, and AUC-ROC.
- Generates a confusion matrix for error analysis.

---

## Results

### Overall Performance
- **Accuracy**: 93.84%
- **F1-Score (Weighted)**: 93.69%

### Classification Report (Sample Metrics):
| Class           | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Battery         | 93.55%    | 95.31% | 94.42%   |
| Biological      | 97.51%    | 98.04% | 97.78%   |
| Plastic         | 92.14%    | 69.72% | 79.38%   |
| Trash           | 88.82%    | 97.57% | 92.99%   |

- **Macro Average**:
  - Precision: 93.89%
  - Recall: 93.84%
  - F1-Score: 93.69%




## References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Dataset source](#)

---

## License
This project is licensed under the MIT License.

