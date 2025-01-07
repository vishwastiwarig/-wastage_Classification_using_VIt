# Wastage Classification using Vision Transformers

## Project Overview
This project aims to classify images of waste types using a Vision Transformer (ViT) model. The notebook leverages the Hugging Face `transformers` and `datasets` libraries for model training and evaluation, focusing on garbage-type image detection using the pretrained model `dima806/garbage_types_image_detection`.

## Features
- **Preprocessing**: Image augmentation and normalization for improved model performance.
- **Model Training**: Fine-tuning the Vision Transformer for image classification.
- **Evaluation**: Detailed metrics including accuracy, F1-score, confusion matrix, and more.
- **Robustness**: Handles corrupted image files gracefully.



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
Provide evaluation results (e.g., accuracy, F1-score) here.



## References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Dataset source](#)



## License
This project is licensed under the MIT License.

