# SignTales: Storytelling for Hearing Impaired Children Using LSTM-based Temporal Model

## Overview
**SignTales** is a storytelling system designed to enhance communication accessibility for deaf and hard-of-hearing children. It converts English text into **stick figure animations** representing **American Sign Language (ASL)** gestures. By leveraging **Natural Language Processing (NLP)** and **deep learning (Seq2Seq LSTM models)**, the system generates smooth, temporally consistent pose animations that bring stories to life in sign language.

This project bridges the gap between textual storytelling and visual sign language, providing an educational and engaging tool for children.

---

## Features
- **Text-to-Gloss Conversion**: Uses SpaCy NLP pipeline to preprocess text and generate gloss sequences.
- **Gloss-to-Pose Translation**: Employs a Seq2Seq model with BiLSTM encoder and LSTM decoder, enhanced by Bahdanau attention.
- **Stick Figure Animation Rendering**: Real-time visualization in both 2D (Canvas) and 3D (Three.js).
- **Custom ASL Dataset**: 572 gesture videos mapped to gloss tokens for training.
- **Web-Deployable**: Lightweight, accessible, and adaptable to any text input within the trained vocabulary.

---

## Methodology
The system follows a **seven-stage pipeline**:

1. **Data Collection & Preparation**  
   - ASL gesture dataset (572 videos)  
   - Domain-specific text corpus for storytelling  

2. **Text Preprocessing & Gloss Generation**  
   - Tokenization, stopword removal, lemmatization  
   - ASL-style grammar ordering (Time → Subject → Noun → Verb → Object → Negation)  

3. **Video Data Preprocessing**  
   - Frame extraction via OpenCV  
   - Landmark detection using MediaPipe Holistic  
   - Normalization, smoothing, and NPZ conversion  

4. **Model Training**  
   - Seq2Seq BiLSTM encoder + LSTM decoder  
   - Attention mechanism for gloss alignment  
   - Teacher forcing with scheduled reduction  
   - Multi-component loss (position, velocity, acceleration, jerk, stop)  

5. **Model Tuning**  
   - Optimized hyperparameters (hidden sizes, dropout, learning rates)  
   - Stop condition tuning for realistic animations  

6. **Model Integration**  
   - REST API backend for inference  
   - Normalized joint coordinates for rendering  

7. **Pose Animation Rendering**  
   - Dual rendering modes (2D & 3D)  
   - Real-time playback with user controls  

---

## Results
- **Final Generation Error**: 0.1834  
- **Velocity Error**: 0.0674  
- **Acceleration Error**: 0.0578  
- **Training Loss**: 0.1080  
- **Validation Loss**: 0.0494  

The system produces **smooth, temporally consistent animations** suitable for storytelling, outperforming static-image or pre-recorded video approaches.

---

## Applications
- **Educational Tools**: Interactive storytelling for deaf children.  
- **Accessibility Solutions**: Bridging communication gaps between signers and non-signers.  
- **Research & Development**: Foundation for future sign language animation systems.  

---

## Limitations
- Vocabulary limited to **71 glosses**.  
- No facial expression generation (critical for full ASL grammar).  
- Stick figure rendering lacks realism compared to full avatars.  

---

## Future Work
- Expand gesture dataset for broader vocabulary.  
- Integrate **facial landmark generation**.  
- Replace stick figures with expressive **3D avatars** (e.g., Blender-based).  
- Extend support to **Nepali Sign Language (NSL)** and other sign languages.  

---

## Tech Stack
- **Languages**: Python  
- **Libraries**: SpaCy, Pandas, OpenCV, MediaPipe, PyTorch, NumPy  
- **Rendering**: Three.js (3D), Canvas (2D)  
- **Model**: Seq2Seq BiLSTM Encoder + LSTM Decoder with Attention  

---


---

## 👩‍💻 Authors
- Bidisha Amatya  
- Prasanna Shakya  
- Prinska Maharjan  
- Sajal Maharjan  
- Ashok GM  

Affiliation: Himalaya College of Engineering, Tribhuvan University, Nepal  

---

## 📜 Citation
If you use this project in your research, please cite:

> Amatya, B., Shakya, P., Maharjan, P., Maharjan, S., & GM, A. (2026). *SignTales: Storytelling for Hearing Impaired Children Using LSTM-based Temporal Model*. Journal of Himalaya College of Engineering, Vol. 3, Issue 1.

---

