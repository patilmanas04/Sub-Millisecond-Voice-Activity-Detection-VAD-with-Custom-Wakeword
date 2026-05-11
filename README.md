# Sub-Millisecond Custom Wakeword Engine

A lightweight, edge-optimized Voice Activity Detection (VAD) and custom wakeword recognition system built from scratch in PyTorch. This model is trained to recognize the specific phrase **"Hello Manas"** in real-time with sub-millisecond inference speeds.

## 📄 Project Overview

The goal of this project was to understand the end-to-end pipeline of audio machine learning, from raw waveform processing to edge-device deployment. Instead of relying on massive, slow APIs, this project implements a custom **Broadcasted Residual Network (BC-ResNet)** designed to run instantly on low-power devices.

### Key Features

- **Sub-Millisecond Inference:** Optimized architecture allowing for real-time processing (< 1ms per frame).
- **Edge-Optimized:** The entire neural network contains only **77,169 trainable parameters**, making it lightweight enough to run on basic CPUs or edge IoT devices without draining resources.
- **Custom Focal Loss:** Handles massive dataset imbalances by mathematically forcing the model to hyper-fixate on hard-to-learn acoustic patterns rather than taking shortcuts with background noise.
- **Push-to-Talk Live Testing:** Includes a synchronous, rolling-buffer inference script to test the model dynamically with a live microphone.

---

## 🧠 Model Architecture

The "brain" of the engine is a custom Convolutional Neural Network (CNN) that processes audio visually.

1. **Feature Extraction:** Raw audio (16kHz) is converted into Log-Mel Spectrograms using `librosa`.
   - `N_MELS`: 40
   - `N_FFT`: 480
   - `HOP_LENGTH`: 160
   - **Input Shape:** `[1, 40, 201]` (1 Channel, 40 Frequency Bins, 201 Time Steps).
2. **The Stem:** A `3x3` 16-channel convolutional layer that acts as the "eyes," extracting basic edges and frequencies.
3. **ResNet Blocks:** 3 stacked `AudioResNetBlock` layers with shortcut connections to prevent vanishing gradients while shrinking the temporal dimensions and deepening the feature maps (up to 64 channels).
4. **Adaptive Pooling:** Crushes the 2D spatial features into a 1D tensor, drastically reducing the parameter count.
5. **Linear Classifier:** A dense layer that outputs a single logit (probability) indicating the presence of the wakeword.

---

## 📊 The Dataset

The model was trained on a custom, heavily augmented dataset of approximately **43,000 audio samples** (2.0 seconds each).

- **Positives (~5,000 samples):** Variations of the wakeword spoken in different tones, speeds, and pitches.
- **Negatives (~38,000 samples):** \* Background noise (MUSAN dataset: static, birds, office chatter, television).
  - Google Speech Commands (Only a couple of words samples were choosen to balance the number of positive samples, the choosen samples are as follows: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go").
  - Random non-target speech.

---

## ⚙️ Installation & Setup

To run this model locally, you will need Python 3.8+ and a working microphone.

**1. Clone the repository:**

```bash
git clone https://github.com/patilmanas04/Sub-Millisecond-Voice-Activity-Detection-VAD-with-Custom-Wakeword.git
cd Sub-Millisecond-Voice-Activity-Detection-VAD-with-Custom-Wakeword
```

**2. Create and activate a virtual environment:**

# Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

# Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install the dependencies:**

```bash
pip install -r requirements.txt
```

_(Note: If you are running this on an NVIDIA GPU, ensure you install the CUDA-compatible version of PyTorch first)._
