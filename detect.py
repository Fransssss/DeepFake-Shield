# detect.py

import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from utils.efficientnet_model import get_efficientnet_model  # Use EfficientNet

# Transform the image to the format EfficientNet expects
preprocess = transforms.Compose([
    transforms.ToPILImage(),                      # Convert OpenCV image to PIL format
    transforms.Resize((224, 224)),                # Resize to EfficientNet-B0 size
    transforms.ToTensor(),                        # Convert to PyTorch tensor
    transforms.Normalize([0.5]*3, [0.5]*3)        # Normalize to [-1, 1]
])

def predict_frame(model, frame_path, device):
    """
    Predicts if a single frame is deepfake or real.
    """
    frame = cv2.imread(frame_path)                              # Load image from disk
    input_tensor = preprocess(frame).unsqueeze(0).to(device)    # Preprocess and batch it

    with torch.no_grad():                   # Inference only (no gradient needed)
        output = model(input_tensor)        # Get prediction
        prob = output.item()                # Convert to scalar

    return prob

def analyze_video_frames(frames_folder, model_path="model/efficientnet_weights.pth", threshold=0.5):
    """
    Loads model, scans each frame in the folder, and returns average prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_efficientnet_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load saved weights
    model.to(device)
    model.eval()

    frame_scores = []

    # Get list of frame image paths
    frame_paths = sorted([
        os.path.join(frames_folder, f)
        for f in os.listdir(frames_folder)
        if f.endswith(".jpg")
    ])

    for path in frame_paths:
        score = predict_frame(model, path, device)
        frame_scores.append(score)

    # Calculate the average score
    avg_score = np.mean(frame_scores)
    result = "Deepfake" if avg_score > threshold else "Real"

    # Show final decision
    print(f"\nVideo classified as: {result}")
    print(f"Average deepfake probability: {avg_score:.4f}")


if __name__ == "__main__":
    test_folder = "extracted_frames/sample1"  # Change as needed
    analyze_video_frames(test_folder)


