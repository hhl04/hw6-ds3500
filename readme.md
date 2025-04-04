# CapTuring: Text Analysis Framework

A modular framework for comparative text analysis designed to analyze and visualize relationships between documents, with a focus on political spectrum positioning and document similarity.

## Overview

CapTuring is an NLP framework that allows you to:

- Compare multiple documents using TF-IDF vectorization and cosine similarity
- Visualize documents' positions on a political spectrum
- Analyze different topics (CCP, Roe v. Wade, etc.) with minimal code changes
- Identify similarities between human-authored baseline documents and LLM-generated content
- Create visualizations including similarity heatmaps, spectrum positioning, and word frequency charts

## Project Structure

- `investigation_3_new_modularized/` - The current, most advanced version of the codebase
  - `CapTuring.py` - The main class file containing the NLP framework
  - `CapTuring_app.py` - The application file that uses the CapTuring class
- `investigation_2_new_modularized/` - Previous iteration of the modularized codebase
- `investigation_1_old_ccp/` - Initial implementation focused on CCP topic
- `investigation_0_old_mvp/` - Minimum viable product version
- `ideation/` - Early project planning and ideas
- `starter_code/` - Initial code templates
- `shourya-test/` - Testing directory

## Setup and Installation

### Prerequisites

```
pip install numpy matplotlib scikit-learn
```

### Getting Started

1. Clone this repository
2. Navigate to the `investigation_3_new_modularized` directory
3. Run the analysis with `python CapTuring_app.py`

## Usage

The framework is designed to analyze documents across a political spectrum for different topics. To use it:

1. Set up your document structure (see below)
2. Modify the `TOPIC` variable in `CapTuring_app.py` to choose which topic to analyze
3. Run the application

## Document Structure

Documents should be organized in the following structure:

```
documents/
  - ccp/                         # Topic folder
    - human/                     # Human-authored baseline documents
      - left/                    # Political perspective
        - left-1.txt             # Document
      - center/
        - center-1.txt
      - right/
        - right-1.txt
    - llm/                       # LLM-generated documents
      - claude/                  # LLM source
        - claude-1.txt           # Document
        - claude-2.txt
      - deepseek/
        - deepseek-1.txt
        - deepseek-2.txt
      - grok/
        - grok-1.txt
        - grok-2.txt
    - support/                   # Support files for this topic
      - stop_words.txt           # Topic-specific stop words
  - roe_vs_wade/                 # Another topic
    - human/
      - ...
    - llm/
      - ...
    - support/
      - ...
```

## Adding New Topics

To add a new topic for analysis:

1. Create a new folder in the `documents/` directory with your topic name (e.g., `climate_change`)
2. Inside this folder, create the following structure:
   - `human/` directory with subdirectories for each political perspective (e.g., `left`, `center`, `right`)
   - `llm/` directory with subdirectories for each LLM source (e.g., `claude`, `grok`)
   - `support/` directory for support files
3. Add at least one text document to each perspective folder in `human/`
4. Add LLM-generated documents to their respective folders in `llm/`
5. Optionally add a `stop_words.txt` file in the `support/` directory
6. Change the `TOPIC` variable in `CapTuring_app.py` to your new topic name

Example of adding a new "gun_control" topic:

```bash
mkdir -p documents/gun_control/human/{left,center,right}
mkdir -p documents/gun_control/llm/{claude,grok,deepseek,openai}
mkdir -p documents/gun_control/support

# Add your text files to the appropriate directories
cp left_gun_article.txt documents/gun_control/human/left/left-1.txt
cp center_gun_article.txt documents/gun_control/human/center/center-1.txt
# ... and so on

# Optional: Add stop words
cp gun_stop_words.txt documents/gun_control/support/stop_words.txt

# Then modify CapTuring_app.py to use this topic
# Change: TOPIC = "ccp" to TOPIC = "gun_control"
```

## Visualizations

The framework generates several visualizations:

1. **Similarity Heatmap**: Shows cosine similarity between all documents
2. **Political Spectrum Positioning**: Shows where each document falls on the political spectrum
3. **Radar Plot** (for 3+ perspectives): Multi-dimensional visualization of document positioning
4. **Word Frequency Charts**: Most common words in each document (with and without stop words)

## Contributing

Feel free to fork this repository and submit pull requests for improvements.
