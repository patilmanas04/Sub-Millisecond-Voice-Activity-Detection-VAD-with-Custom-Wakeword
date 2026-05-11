import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
import time
import librosa.display
import matplotlib.pyplot as plt

# The Neural Network Architecture
class AudioResNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1=nn.BatchNorm2d(out_channels)
    self.relu=nn.ReLU(inplace=True)
    self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2=nn.BatchNorm2d(out_channels)

    self.shortcut=nn.Sequential()
    if stride!=1 or in_channels!=out_channels:
      self.shortcut=nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    residual=self.shortcut(x)
    out=self.conv1(x)
    out=self.bn1(out)
    out=self.relu(out)
    out=self.conv2(out)
    out=self.bn2(out)
    out+=residual
    out=self.relu(out)

    return out

class WakeWordBrain(nn.Module):
  def __init__(self):
    super().__init__()

    self.stem=nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True)
    )
    self.layer1=AudioResNetBlock(16, 16, stride=1)
    self.layer2=AudioResNetBlock(16, 32, stride=2)
    self.layer3=AudioResNetBlock(32, 64, stride=2)
    self.adaptive_pool=nn.AdaptiveAvgPool2d((1, 1))
    self.classifier=nn.Linear(64, 1)

  def forward(self, x):
    x=self.stem(x)
    x=self.layer1(x)
    x=self.layer2(x)
    x=self.layer3(x)
    x=self.adaptive_pool(x)
    x=torch.flatten(x, 1)
    x=self.classifier(x)
    return x

# --- CONFIGURATION & SETUP ---
SR = 16000
DURATION = 2.0
TARGET_SAMPLES = int(SR * DURATION)

# Spectrogram Params 
N_MELS = 40
N_FFT = 480
HOP_LENGTH = 160

CONFIDENCE_THRESHOLD = 0.98
# Load the weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Booting up brain on: {device}...")

model = WakeWordBrain().to(device)
model.load_state_dict(torch.load("best_wakeword_brain.pth", map_location=device))
model.eval() # Sets model to inference mode

print("\n▶️  PUSH-TO-TALK WAKEWORD ENGINE ACTIVE")
print("Press [ENTER] to record for 2 seconds. Type 'q' and press [ENTER] to quit.\n")

# --- THE PUSH-TO-TALK LOOP ---
while True:
    user_input = input("🎙️  Ready. Press ENTER to start listening... ")

    if user_input.lower() == 'q':
        print("🛑  Engine Shut Down.")
        break

    print("🔴  Listening for 2 seconds... (Say 'Hello Manas')")

    # Recording 2 seconds of audio clip
    recording = sd.rec(TARGET_SAMPLES, samplerate=SR, channels=1, blocking=True)
    
    # Flatten from shape (32000, 1) to (32000,)
    audio_data = recording[:, 0]

    # Create the image (Mel Spectrogram)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    
    # Conver the mel spectrogram to decibels
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # --- SPECTROGRAM VISUALIZATION ---
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(log_mel_spec, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='magma')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title("Live Brain Input: What the AI Sees")
    # plt.tight_layout()
    # plt.show(block=False) 
    # plt.pause(2.5)
    # plt.close()
    
    tensor_img = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Pass it to the trained neural network
    with torch.no_grad():
        start_time = time.time()
        output = model(tensor_img)
        prob = torch.sigmoid(output).item() 
        infer_time = (time.time() - start_time) * 1000 
        
    if prob >= CONFIDENCE_THRESHOLD:
        print(f"✅ HELLO MANAS DETECTED! (Confidence: {prob*100:.1f}% | Inference Speed: {infer_time:.2f}ms)\n")
    else:
        print(f"❌ Ignored. (Confidence: {prob*100:.1f}%) - Background Noise or Wrong Word.\n")