# DeepFake Shield : AI-Powered Deepfake Video Detector

## Overview

**DeepFake Shield** is a socially driven AI application built to detect whether a video has been synthetically altered using deepfake technologies. This tool is part of a broader mission to fight digital abuse, particularly the harmful rise of AI-generated pornographic content that misuses the likeness of innocent individuals. The app offers researchers, journalists, content moderators, and the general public a reliable way to verify the authenticity of videos.

This project is ongoing and publicly documented on LinkedIn to inspire transparency, accountability, and ethical AI development.

---

## Features

* Detects AI-generated deepfake content in videos
* Uses frame-based image analysis with advanced neural networks
* Supports full video analysis with per-frame confidence scoring
* Outputs an explainable result (confidence score, key frames)
* Optionally generates a downloadable PDF report for evidence or documentation
* Web-based interface for easy use (Streamlit or Gradio)
* "Safe Mode" to blur explicit content automatically
* Victim-report feature (planned)

---

## Motivation

The abuse of deepfake technology in the creation of fake pornographic content is not only unethical but criminal in nature. Innocent individuals—often women—have had their faces superimposed into explicit videos without consent. This application is a contribution to the fight against this growing digital threat.

---

## Model Architecture

We use a two-stage model approach:

### Stage 1: Frame Extraction

* Videos are decomposed into key frames using OpenCV.
* Each frame is preprocessed (face detected and aligned if possible).

### Stage 2: Deepfake Detection

* **Initial Model**: XceptionNet (pretrained on FaceForensics++)
* **Advanced Models** (optional): EfficientNet, TimeSformer, ViT
* Output: For each frame, a probability of being a deepfake is predicted.
* Aggregation: Video-level score is averaged from all frame predictions.

---

## Roadmap

| Phase   | Description                                                 |
| ------- | ----------------------------------------------------------- |
| Phase 1 | Dataset setup and frame extraction module                   |
| Phase 2 | Train & test baseline deepfake detection model              |
| Phase 3 | Build MVP with basic web UI for video uploads               |
| Phase 4 | Add explainability (e.g., Grad-CAM overlays)                |
| Phase 5 | Add NSFW filter, Safe Mode, and PDF reporting               |
| Phase 6 | Optimize and deploy at scale (Docker + Render/AWS)          |
| Phase 7 | Add reporting tools, feedback loop, and case study showcase |

---

## How It Works

1. **Upload a Video**
2. **Extract Frames** (1 every few seconds or based on motion)
3. **Run Frame Analysis** using pretrained deepfake classifier
4. **Display Results**: Each frame gets a deepfake confidence score
5. **Decision**: Aggregated score determines if video is flagged
6. **Download Report** (optional): Summary with findings and evidence

---

## Getting Started

### Requirements

* Python 3.8+
* pip or conda

### Installation

```bash
git clone https://github.com/yourusername/deepfake-shield.git
cd deepfake-shield
pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
# or
streamlit run app.py
```

### Run Tests

```bash
pytest tests/
```

---

## Directory Structure

```
deepfake-shield/
│
├── app.py                  # Web interface
├── detect.py               # Core logic for detection
├── extractor.py            # Frame extraction
├── model/                  # Pretrained and fine-tuned models
├── utils/                  # Helper functions
├── reports/                # Generated PDF reports
├── static/                 # Uploaded videos and frames
├── requirements.txt        # Required libraries
└── README.md               # This file
```

---

## Model Training (Optional)

To train or fine-tune your own model:

1. Prepare dataset (e.g., FaceForensics++, DFDC)
2. Extract frames and labels
3. Use `train.py` script to start training
4. Save weights to `model/` folder

---

## Future Improvements

* Real-time webcam detection
* Multi-modal analysis (voice + visual cues)
* Blockchain watermarking for verified content
* Integration with victim-report support groups

---

## Use Cases

* Content moderation for social platforms
* Journalism and media authentication
* Legal evidence preparation
* Personal safety and verification for individuals

---

## Ethical Considerations

* All models are trained on publicly available and ethically sourced datasets.
* The tool is designed to support victims and empower safety.
* No explicit content is displayed. Safe Mode is on by default.
* We do not store videos or personally identifiable data.

---

## Contributing

If you'd like to contribute, please fork the repo and submit a pull request. Open issues for bugs, suggestions, or ideas. Community involvement is encouraged.

---

## License

This project is open source under the MIT License. Please Use it freely, but ethically.

---

## Contact

Creator: Fransiskus Agapa

LinkedIn: https://www.linkedin.com/in/fransiskus-agapa/

---