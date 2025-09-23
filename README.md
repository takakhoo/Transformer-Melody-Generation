# Transformer Melody Generation

This project implements a transformer-based melody generation system using deep learning techniques.

## Project Structure

- `__init__.py` - Package initialization
- `melodygenerator.py` - Main melody generation logic
- `melodypreprocessor.py` - Data preprocessing for melodies
- `train.py` - Training script for the transformer model
- `transformer.py` - Transformer architecture implementation

## Setup

1. Create a conda environment:
   ```bash
   conda create -n MusicAIPractice python=3.8 -y
   conda activate MusicAIPractice
   ```

2. Install dependencies:
   ```bash
   pip install music21==8.3.0 tensorflow==2.13.0
   ```

## Usage

Run the training script:
```bash
python train.py
```

Generate melodies:
```bash
python melodygenerator.py
```

## Learning Resources

Based on the Generative Music AI Course by The Sound of AI.
