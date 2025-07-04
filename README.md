# Chest X-Ray Pathology Detection

This repository contains Jupyter notebooks for converting a chest X-ray dataset to the YOLO format and training a YOLOv8 model to detect lung pathologies.

## Repository Structure

- `1_analise_dataset.ipynb` – Dataset analysis and visualization.
- `2_conversao_yolo.ipynb` – Converts annotations from `train.json` and `test.json` to YOLO format.
- `3_treino.ipynb` – Training notebook using Ultralytics YOLOv8.
- `4_eval.ipynb` – Model evaluation notebook.
- `5_detecao.ipynb` – Example of running inference on new images.
- `train.json`, `test.json` – Original dataset annotations.

## Setup

Install dependencies in a virtual environment and start Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

Each notebook is designed to run sequentially:
1. Explore and analyze the dataset.
2. Convert annotations to YOLO format.
3. Train a YOLOv8 detector.
4. Evaluate the trained model.
5. Run detection on sample images.

## Notes

Training requires a GPU for best performance. Adjust paths in the notebooks to match your data locations before running.
