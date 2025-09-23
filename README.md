# 🎵 Transformer Melody Generation

A sophisticated deep learning system that generates musical melodies using state-of-the-art Transformer architecture. This project implements a complete end-to-end pipeline for training and generating melodies using sequence-to-sequence learning with attention mechanisms.

## 🎯 Overview

This system leverages the power of Transformers - the same architecture that revolutionized natural language processing - to understand and generate musical sequences. By treating melodies as sequences of musical tokens (notes with pitch and duration), the model learns complex musical patterns and can generate novel, coherent melodies.

## 🏗️ Architecture Deep Dive

### Core Components

#### 1. **Transformer Model (`transformer.py`)**
The heart of the system implements a complete Transformer architecture with:

- **Encoder-Decoder Architecture**: Processes input sequences and generates output sequences
- **Multi-Head Attention**: Allows the model to focus on different parts of the input simultaneously
- **Positional Encoding**: Provides temporal information using sinusoidal functions
- **Layer Normalization**: Stabilizes training and improves convergence
- **Feed-Forward Networks**: Adds non-linear transformations between attention layers

**Key Innovation**: The model uses sinusoidal positional encoding to understand the temporal relationships between musical notes, crucial for maintaining musical coherence.

#### 2. **Melody Preprocessor (`melodypreprocessor.py`)**
Intelligent data preparation system that:

- **Tokenizes Musical Sequences**: Converts musical notes into numerical tokens
- **Dynamic Sequence Padding**: Handles variable-length melodies efficiently
- **Input-Target Pair Generation**: Creates training pairs by shifting sequences
- **TensorFlow Dataset Creation**: Optimizes data loading for GPU training

**Smart Feature**: Automatically calculates maximum sequence length from the dataset, ensuring optimal memory usage.

#### 3. **Melody Generator (`melodygenerator.py`)**
Advanced inference engine that:

- **Iterative Generation**: Builds melodies note-by-note using autoregressive prediction
- **Context-Aware Prediction**: Uses the entire generated sequence as context for next note
- **Token Decoding**: Converts numerical predictions back to musical notation
- **Length Control**: Generates melodies up to specified maximum length

**Innovation**: Uses teacher forcing during training but autoregressive generation during inference for optimal performance.

#### 4. **Training Pipeline (`train.py`)**
Sophisticated training system featuring:

- **Custom Loss Function**: Implements masked sparse categorical crossentropy
- **Gradient Tape**: Enables automatic differentiation for backpropagation
- **Adam Optimizer**: Uses adaptive learning rates for stable training
- **Batch Processing**: Efficiently processes multiple melodies simultaneously

**Advanced Feature**: Implements padding masks to ignore loss on padded positions, focusing learning on actual musical content.

## 🎼 Musical Representation

The system uses a sophisticated musical tokenization scheme:

```
Format: [PITCH][OCTAVE]-[DURATION]
Examples: C4-1.0, G4-2.0, A4-0.5
```

- **Pitch**: Musical note (C, D, E, F, G, A, B)
- **Octave**: Octave number (4 = middle octave)
- **Duration**: Note length in quarter notes (1.0 = quarter note, 2.0 = half note)

## 🚀 Technical Highlights

### Advanced Features

1. **Attention Mechanisms**: Multi-head attention allows the model to understand complex musical relationships
2. **Positional Encoding**: Sinusoidal encoding preserves temporal information in musical sequences
3. **Dynamic Padding**: Efficiently handles melodies of varying lengths
4. **Masked Training**: Focuses learning on actual musical content, not padding
5. **Autoregressive Generation**: Creates coherent melodies through iterative prediction

### Performance Optimizations

- **TensorFlow Integration**: Leverages GPU acceleration for fast training
- **Batch Processing**: Processes multiple melodies simultaneously
- **Efficient Tokenization**: Optimized text processing for musical sequences
- **Memory Management**: Dynamic sequence length calculation reduces memory usage

## 📊 Dataset Structure

The system works with JSON datasets containing melody sequences:

```json
[
  "C4-1.0, C4-1.0, G4-1.0, G4-1.0, A4-1.0, A4-1.0, G4-2.0",
  "E4-1.0, D4-1.0, C4-1.0, D4-1.0, E4-1.0, E4-1.0, E4-2.0"
]
```

Each melody is represented as a comma-separated string of musical tokens.

## 🎮 Usage

### Training the Model
```bash
python train.py
```

### Generating Melodies
```python
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor

# Load trained model and tokenizer
generator = MelodyGenerator(trained_transformer, tokenizer)

# Generate melody from seed sequence
start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0"]
new_melody = generator.generate(start_sequence)
```

## 🔧 Configuration

### Model Parameters
- **Layers**: 2 (Encoder + Decoder layers)
- **Model Dimension**: 64
- **Attention Heads**: 2
- **Feed-Forward Dimension**: 128
- **Dropout Rate**: 0.1
- **Max Positions**: 100

### Training Parameters
- **Epochs**: 10
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

## 🎵 Musical Capabilities

The system can:
- **Generate Coherent Melodies**: Creates musically sensible note sequences
- **Learn Musical Patterns**: Understands common melodic progressions
- **Maintain Temporal Structure**: Preserves rhythm and timing relationships
- **Handle Variable Lengths**: Generates melodies of different lengths
- **Preserve Musical Context**: Uses attention to maintain melodic coherence

## 🚀 Future Enhancements

Potential improvements include:
- **Multi-Track Generation**: Generate harmonies and accompaniments
- **Style Transfer**: Learn different musical styles
- **Real-Time Generation**: Interactive melody creation
- **Advanced Masking**: Implement look-ahead and padding masks
- **Music21 Integration**: Export to MIDI and sheet music formats

## 🎓 Educational Value

This project demonstrates:
- **Deep Learning Architecture**: Complete Transformer implementation
- **Sequence-to-Sequence Learning**: Advanced NLP techniques applied to music
- **Attention Mechanisms**: Understanding of modern AI architectures
- **Music Information Retrieval**: Bridging music theory and machine learning
- **End-to-End Pipeline**: Complete ML system from data to generation

## 📚 Learning Resources

Based on the Generative Music AI Course by The Sound of AI, this implementation showcases cutting-edge techniques in:
- Neural sequence modeling
- Attention-based architectures
- Music information processing
- Deep learning for creative applications

---

*This system represents a sophisticated application of modern AI to the creative domain, demonstrating how advanced machine learning techniques can be adapted for musical generation and understanding.*
