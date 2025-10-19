# 🩺 AISTETHOSCOPE: AI-Powered Heart Sound Classification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

A comprehensive deep learning system for automated heart sound analysis and classification using advanced neural network architectures including LSTM, CNN-LSTM hybrid, and Transformer models.

## 🎯 Overview

AISTETHOSCOPE is an AI-powered medical diagnostic tool that can automatically classify heart sounds into different categories, helping healthcare professionals identify potential cardiac conditions. The system processes audio recordings of heart sounds and provides accurate classification using state-of-the-art deep learning techniques.

### Key Features

- **Multi-Class Classification**: Identifies 6 different heart sound categories
- **Advanced Deep Learning**: Implements LSTM, CNN-LSTM, and Transformer architectures
- **Comprehensive Audio Processing**: Uses MFCC features, spectral analysis, and temporal information
- **Data Augmentation**: Multiple augmentation techniques for improved model robustness
- **Medical-Grade Analysis**: Based on clinical heart sound characteristics

## 🏥 Heart Sound Categories

The system can classify heart sounds into the following categories:

| Category | Description | Clinical Significance |
|----------|-------------|----------------------|
| **Normal** | Healthy heart sounds with clear "lub-dub" pattern | No cardiac abnormalities |
| **Murmur** | Turbulent blood flow sounds between heartbeats | Potential valve disorders |
| **Artifact** | Background noise, feedback, or non-heart sounds | Recording quality issues |
| **Extrasystole** | Extra or skipped heartbeats | Arrhythmia detection |
| **Extra Heart Sounds** | Additional sounds beyond normal S1/S2 | Potential cardiac conditions |
| **Extrahls** | Extra heart sounds in specific patterns | Clinical abnormality indicators |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.19.0
- Keras 3.10.0
- Librosa
- NumPy, Pandas, Matplotlib
- Scikit-learn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AISTETHOSCOPE.git
   cd AISTETHOSCOPE
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main notebook**
   ```bash
   jupyter notebook AISTETHOSCOPE.ipynb
   ```

### Quick Usage

```python
# Load and preprocess heart sound data
from AISTETHOSCOPE import HeartSoundClassifier

# Initialize the classifier
classifier = HeartSoundClassifier()

# Load your heart sound audio file
audio_file = "path/to/heart_sound.wav"

# Classify the heart sound
prediction = classifier.predict(audio_file)
print(f"Predicted category: {prediction}")
```

## 📊 Dataset Information

### Dataset Statistics
- **Total Samples**: 832 heart sound recordings
- **Classes**: 6 categories
- **Audio Format**: WAV files
- **Duration**: 1-30 seconds per recording
- **Sampling Rate**: 16 kHz (processed)

### Data Distribution
```
Normal:     351 samples (42.2%)
Murmur:     241 samples (29.0%)
Artifact:   126 samples (15.1%)
Extrasystole: 19 samples (2.3%)
Extra Heart Sounds: 95 samples (11.4%)
```

## 🧠 Model Architectures

### 1. LSTM Model
- **Architecture**: 2-layer LSTM with dropout
- **Features**: 40-dimensional MFCC vectors
- **Performance**: 53-70% accuracy

### 2. CNN-LSTM Hybrid
- **Architecture**: Convolutional layers + LSTM
- **Advantages**: Spatial and temporal feature extraction
- **Use Case**: Complex pattern recognition

### 3. Transformer Model
- **Architecture**: Multi-head attention mechanism
- **Advantages**: Long-range dependency modeling
- **Performance**: Improved accuracy over LSTM

### 4. Attention LSTM
- **Architecture**: LSTM with attention mechanism
- **Advantages**: Focus on important temporal features
- **Use Case**: Critical heart sound pattern detection

## 🔧 Technical Implementation

### Audio Processing Pipeline

1. **Audio Loading**: Load WAV files using Librosa
2. **Preprocessing**: Normalize and pad/truncate to fixed length
3. **Feature Extraction**: 
   - MFCC (Mel-Frequency Cepstral Coefficients)
   - Spectral features (centroid, rolloff, bandwidth)
   - Temporal features (zero-crossing rate, RMS energy)
4. **Data Augmentation**: Noise injection, pitch shifting, speed perturbation

### Feature Engineering

```python
# Enhanced feature extraction
def extract_enhanced_features(audio_file, sr=16000, duration=12):
    # Load audio
    y, sr = librosa.load(audio_file, sr=sr, duration=duration)
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    # Temporal features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    return combined_features
```

### Model Training

```python
# Model compilation and training
model.compile(
    optimizer='Adamax',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training with callbacks
callbacks = [
    EarlyStopping(patience=12),
    ReduceLROnPlateau(patience=12),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

## 📈 Performance Metrics

### Current Results

| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------|------------------|-------------------|---------------|
| LSTM | 61% | 53% | 63% |
| CNN-LSTM | 68% | 58% | 65% |
| Transformer | 72% | 64% | 69% |
| Attention LSTM | 70% | 61% | 67% |

### Evaluation Metrics

- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Precision, recall, F1-score per class
- **ROC Curves**: Receiver Operating Characteristic analysis
- **Cross-Validation**: K-fold validation for robust evaluation

## 🛠 Advanced Features

### Data Augmentation

```python
# Audio augmentation techniques
def augment_audio(y, sr, augmentation_type='noise'):
    if augmentation_type == 'noise':
        noise_factor = 0.005
        noise = np.random.normal(0, noise_factor, len(y))
        return y + noise
    elif augmentation_type == 'pitch':
        steps = np.random.randint(-2, 3)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    # ... more augmentation types
```

### Ensemble Methods

- **Voting Classifier**: Combines multiple model predictions
- **Stacking**: Meta-learning approach for improved accuracy
- **Bagging**: Bootstrap aggregating for robust predictions

### Model Optimization

- **Hyperparameter Tuning**: Grid search and random search
- **Architecture Search**: Neural architecture optimization
- **Transfer Learning**: Pre-trained model fine-tuning

## 📁 Project Structure

```
AISTETHOSCOPE/
├── AISTETHOSCOPE.ipynb          # Main analysis notebook
├── CREWAIAGENTS.py              # AI agent implementations
├── movie_qa_demo.py             # Demo application
├── INSTALLATION_GUIDE.md        # Installation instructions
├── new_blog-post.md             # Project documentation
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Dataset directory
│   ├── set_a/                   # Training set A
│   ├── set_b/                   # Training set B
│   └── *.csv                    # Metadata files
├── models/                      # Saved model files
│   ├── best_model.keras
│   ├── transformer_model.keras
│   └── ensemble_models/
├── src/                         # Source code
│   ├── audio_processing.py      # Audio preprocessing
│   ├── feature_extraction.py    # Feature engineering
│   ├── models.py                # Model definitions
│   └── utils.py                 # Utility functions
└── tests/                       # Test files
    ├── test_audio_processing.py
    ├── test_models.py
    └── test_utils.py
```

## 🔬 Research & Development

### Current Research Areas

1. **Transformer Architecture**: Exploring attention mechanisms for audio
2. **Data Augmentation**: Advanced augmentation techniques
3. **Transfer Learning**: Pre-trained model adaptation
4. **Real-time Processing**: Streaming audio analysis
5. **Mobile Deployment**: Edge device optimization

### Future Enhancements

- [ ] Real-time audio processing capabilities
- [ ] Mobile app development
- [ ] Integration with electronic health records
- [ ] Advanced ensemble methods
- [ ] Interpretability and explainability features
- [ ] Multi-modal analysis (ECG + audio)

## 🏥 Medical Applications

### Clinical Use Cases

- **Primary Care**: Initial heart sound screening
- **Telemedicine**: Remote cardiac assessment
- **Medical Education**: Training and simulation
- **Research**: Cardiovascular disease studies
- **Emergency Medicine**: Rapid cardiac evaluation

### Regulatory Considerations

⚠️ **Important**: This system is for research and educational purposes only. It should not be used as a primary diagnostic tool without proper clinical validation and regulatory approval.

## 🤝 Contributing

We welcome contributions to improve AISTETHOSCOPE! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/AISTETHOSCOPE.git
cd AISTETHOSCOPE
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

## 📚 Documentation

- [Installation Guide](INSTALLATION_GUIDE.md)
- [API Documentation](docs/api.md)
- [Model Architecture Details](docs/models.md)
- [Performance Benchmarks](docs/benchmarks.md)
- [Medical Validation](docs/medical_validation.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Heart Sound Challenge dataset
- **Libraries**: TensorFlow, Keras, Librosa, Scikit-learn
- **Research**: Medical literature on heart sound analysis
- **Community**: Open source contributors and medical professionals

## 📞 Contact

- **Project Maintainer**: [Your Name](mailto:your.email@example.com)
- **Project Link**: [https://github.com/yourusername/AISTETHOSCOPE](https://github.com/yourusername/AISTETHOSCOPE)
- **Issues**: [GitHub Issues](https://github.com/yourusername/AISTETHOSCOPE/issues)

## 📊 Citation

If you use AISTETHOSCOPE in your research, please cite:

```bibtex
@software{aistethoscope2024,
  title={AISTETHOSCOPE: AI-Powered Heart Sound Classification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/AISTETHOSCOPE},
  note={Deep learning system for automated heart sound analysis}
}
```

---

**⭐ Star this repository if you find it helpful!**

*Built with ❤️ for advancing medical AI and improving healthcare outcomes.*
