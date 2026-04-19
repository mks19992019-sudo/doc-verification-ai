# CNN Training (ResNet-50 Fine-tuning)

## Dataset Structure

```
dataset/
├── train/
│   ├── genuine/   ← genuine document images
│   └── forged/   ← forged/manipulated document images
└── val/
    ├── genuine/
    └── forged/
```

## Run on Kaggle

1. Upload your `dataset/` folder to Kaggle (as a dataset)
2. Create a new Kaggle Notebook with **GPU** (P100 or T4)
3. Install dependencies:
   ```python
   !pip install torch torchvision scikit-learn Pillow tqdm
   ```
4. Upload `train.py` to the notebook or copy-paste the code
5. Run:
   ```python
   !python train.py --data /kaggle/input/your-dataset --epochs 10 --batch_size 16 --lr 1e-4
   ```
6. Download `weights/resnet50_forgery.pt` from the output
7. Copy it to `forgery-detection/weights/resnet50_forgery.pt`

## Local Run (with GPU)

```bash
cd training
pip install -r requirements.txt
python train.py --data ../dataset --epochs 10 --batch_size 16
```

## Output

Training saves `weights/resnet50_forgery.pt` — drop this file into `forgery-detection/weights/` and the CNN model will automatically activate in the backend pipeline.
