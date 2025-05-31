# use_model.py

import os
import sys

MODEL_PATH = "model/efficientnet_weights.pth"

# Check if trained model exists before using
if not os.path.exists(MODEL_PATH):
    print(" Model not found at 'model/efficientnet_weights.pth'.")
    print(" Please train the model first using 'train_model_colab.ipynb' in Google Colab.")
    sys.exit()

# If this runs, you're safe to use the model
print(" Model found. You can now run detection (e.g., detect.py).")
