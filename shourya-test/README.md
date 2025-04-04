# CapTuring NLP Framework

A modularized NLP framework for comparative text analysis designed to analyze documents across a political spectrum. This framework allows you to compare multiple documents (such as LLM-generated articles) against baseline documents (such as human-written articles with different political leanings).

## Project Structure

- `CapTuring.py`: The main class that provides NLP functionality
- `CapTuring_app.py`: Application that demonstrates how to use the CapTuring class
- `documents/`: Directory where you should place your text files for analysis

## Features

- Document loading and text processing
- TF-IDF vectorization for document representation
- Cosine similarity calculation between documents
- Political spectrum analysis based on similarity to baseline documents
- Visualization tools:
  - Similarity heatmap
  - Political spectrum positioning
  - Word importance visualization

## Requirements

The framework requires the following Python packages:

```
numpy
matplotlib
scikit-learn
```

You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

1. Create a `documents` directory and add your text files:
   - Add human-written baseline articles (e.g., `human-conservative-1.txt`, `human-progressive-1.txt`)
   - Add LLM-generated articles or other documents you want to analyze

2. Run the application:

```bash
python CapTuring_app.py
```

3. The application will:
   - Load all documents from the `documents` directory
   - Set baseline documents (human articles representing different political views)
   - Compute similarity between all documents
   - Generate visualizations showing the political leaning of each document

## Extending the Framework

The framework is designed to be extensible. You can:

- Create custom parsers for different document formats
- Add new analysis methods to the CapTuring class
- Create new visualization functions

See the example of a custom JSON parser in `CapTuring_app.py`.

## Example

```python
from CapTuring import CapTuring

# Create an instance
capturing = CapTuring()

# Load documents
capturing.load_document('documents/human-conservative-1.txt', 'conservative')
capturing.load_document('documents/human-progressive-1.txt', 'progressive')
capturing.load_document('documents/llm-article-1.txt', 'llm-1')

# Set baseline documents
capturing.set_baseline_documents(['conservative', 'progressive'])

# Compute similarity
capturing.compute_similarity_matrix()

# Compare LLM article to baselines
results = capturing.compare_to_baselines('llm-1')
print(results)

# Generate visualization
capturing.visualize_political_spectrum()
```