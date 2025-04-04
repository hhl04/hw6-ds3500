# Textastic: Analyzing Political Bias in AI Text Generation

## Overview
This project investigates bias in AI-generated descriptions of the Chinese Communist Party (CCP) using a custom text analysis framework. By comparing LLM outputs (Claude, DeepSeek, OpenAI) against reference sources (Wikipedia, Chinese Government, American Government), we quantify how AI-generated content aligns with different political perspectives.

## Repository Structure

- **CCP Investigation** (main focus): Measures how LLM outputs compare to neutral, pro-China, and anti-China reference texts
- **Starter Code**: Initial framework implementation 
- **Ideation**: Brainstorming materials
- **MVP**: Minimal viable product

## Key Features

- **Custom NLP Implementation**
  - TF-IDF vectorization built from scratch
  - Cosine similarity calculation
  - Stop words filtering
  - Political terminology analysis

- **Visualizations**
  - Similarity heatmaps between LLMs and reference sources
  - Bias distribution charts
  - Term usage analysis

## Usage

```
python textastic_app.py
```

Requires: Python 3.6+, NumPy, Matplotlib

## Files

- `textastic.py`: Core analysis framework
- `textastic_app.py`: Application driver
- `textastic_parsers.py`: Custom document parsers
- `stop_words_ccp.txt`: Stop words list

## Extending
The framework is modular by design - add parsers for new document types or implement additional analysis methods in the Textastic class.