# detect.py
# Purpose: Load a trained EfficientNet model and run deepfake detection on each frame of a video.

import os                          # For handling file paths
import torch                       # PyTorch core
import cv2                         # OpenCV for image reading
import numpy as np                 # For numeric operations like averaging
from torchvision import transforms # To format images for model
from utils.efficientnet_model import get_efficientnet_model  # Load model architecture
from generate_report import generate_report  # Import report generation function

# ================================================
# Define the preprocessing steps for each frame
# EfficientNet expects 224x224 images normalized to [-1, 1]
# ================================================
preprocess = transforms.Compose([
    transforms.ToPILImage(),                      # Convert OpenCV BGR image to PIL RGB
    transforms.Resize((224, 224)),                # Resize to EfficientNet's expected input
    transforms.ToTensor(),                        # Convert to PyTorch tensor
    transforms.Normalize([0.5]*3, [0.5]*3)         # Normalize each channel to [-1, 1]
])

# ================================================
# Predict the deepfake probability for a single frame
# ================================================
def predict_frame(model, frame_path, device):
    """
    Loads and classifies a single frame using the trained model.
    Returns a probability score between 0 (real) and 1 (deepfake).
    """
    frame = cv2.imread(frame_path)                                 # Load image from disk
    input_tensor = preprocess(frame).unsqueeze(0).to(device)       # Apply preprocessing & add batch dimension

    with torch.no_grad():                                          # Disable gradient tracking for inference
        output = model(input_tensor)                               # Run forward pass
        prob = output.item()                                       # Convert model output tensor to Python float

    return prob

# ================================================
# Analyze a folder of frames and return video classification
# ================================================
def analyze_video_frames(frames_folder, model_path="model/efficientnet_weights.pth", threshold=0.5):
    """
    Loads trained model, classifies each frame, and computes an average deepfake score.
    Returns and prints the final verdict for the video.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Load model architecture and weights
    model = get_efficientnet_model(pretrained=False)                        # Load model structure
    model.load_state_dict(torch.load(model_path, map_location=device))     # Load saved weights
    model.to(device)                                                        # Move model to device
    model.eval()                                                            # Set model to inference mode

    frame_scores = []  # List to collect predictions for all frames

    # Get sorted list of image file paths (assumes .jpg format)
    frame_paths = sorted([
        os.path.join(frames_folder, f)
        for f in os.listdir(frames_folder)
        if f.endswith(".jpg")
    ])

    # Loop through each frame and get prediction score
    for path in frame_paths:
        score = predict_frame(model, path, device)
        frame_scores.append(score)

    # Compute average deepfake score across all frames
    avg_score = np.mean(frame_scores)
    result = "Deepfake" if avg_score > threshold else "Real"

    # Display the result
    print(f"\nVideo classified as: {result}")
    print(f"Average deepfake probability: {avg_score*100:.2f} %")

    # Save supportive and friendly report]
    report_path = os.path.join("reports", f"{os.path.basename(frames_folder)}_report.txt")
    generate_report(frame_scores, report_path, video_name=os.path.basename(frames_folder), threshold=threshold)

# ================================================
# Run analysis if this script is run directly
# ================================================
if __name__ == "__main__":
    test_folder = "extracted_frames/sample_1"  # Folder containing video frames
    analyze_video_frames(test_folder)


