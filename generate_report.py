import os 
import numpy as np
from datetime import datetime 

def generate_report(frame_scores, output_path, video_name, threshold=0.5):
    """
    Generates a user-friendly, supportive report based on frame scores.
    Designed for readability and dignity — especially for victims.

    Parameters:
        frame_scores (list): List of deepfake probabilities for each frame.
        output_path (str): Path to save the report.
        video_name (str): Name of the analyzed video.
        threshold (float): Threshold for classifying as deepfake or real.
    """
    
    avg_score = np.mean(frame_scores)
    is_deepfake = avg_score > threshold 

    # Formate the result as human-friendly text
    result = "🔴 Likely AI-Generated (DeepFake)" if is_deepfake else "🟢 Likely Real (Authentic)"
    confidence = f"{avg_score*100:.2f}%"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the report content
    report_text = report_text = f"""\
        DeepFake Shield – Detection Report
        ----------------------------------------------
        Date: {timestamp}
        Video Analyzed: {video_name}
        Detection Result: {result}
        Confidence Score: {confidence}

        Interpretation:
        This result was obtained by analyzing each frame of the video using a neural network trained to recognize deepfake artifacts.
        A confidence score close to 1.00 means the system found high statistical evidence of AI-generated manipulation.

        ------------------------------------------------
        Model Details:
        • Model: EfficientNet-B0
        • Framework: PyTorch
        • Input size: 224x224 RGB frames
        • Output: Binary classification (Real or Deepfake)

        Training Summary:
        • Dataset: Celeb-DF v2 + DeepFakeDetection + RealFace HQ (combined, ~45,000 frames)
        • Training Accuracy: ~97.6%
        • Validation Accuracy: ~93.1%
        • F1 Score: 0.92
        • AUC (Area Under Curve): 0.96

        ------------------------------------------------
        Confidence Interpretation Guide:
        • 0.00–0.30: Very likely real
        • 0.31–0.50: Possibly real
        • 0.51–0.70: Possibly deepfake
        • 0.71–0.90: Likely deepfake
        • 0.91–1.00: Strongly indicates deepfake

        ------------------------------------------------
        Ethical Note:
        This report does not assert legal guilt or innocence. It is a technical indicator intended to support investigation, raise awareness,
        and provide guidance for further action.

        If your likeness appears in a video and this report suggests manipulation, consider speaking with trusted individuals, legal experts,
        mental health professionals, or cybercrime units.

        You are not alone. You are not at fault.
        """

    
    # Save the report to the specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\nYour Report generated successfully: {output_path}\n")


