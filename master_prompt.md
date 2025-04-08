# CapTuring: Analyzing Political Bias in LLMs

## Overview

CapTuring is an NLP framework designed to quantify and visualize political bias in large language model outputs through comparative text analysis. We use methods like cosine similarity on TF-IDF representations to analyze LLM-generated content along a political spectrum defined by human-authored baseline documents.

For example, we first choose a topic such as the CCP, Roe vs. Wade, or Trump's Tariffs. Then, we select human-written documents (usually one each for "left" like CNN, "center" like Wikipedia or AP, and "right" like Fox News) as our baseline documents for comparison. We then ask different LLMs such as Claude, OpenAI, Google Gemini, or Deepseek to write an article (usually two articles per LLM) about the topic.

Then, we're able to compare these LLM outputs to the human-written biased documents to examine their closeness to each, and therefore measure each LLM's "bias".

## Data Structure

```
CapTuring.py              # Core class implementing the NLP framework
CapTuring_app.py          # Application layer for topic selection and console updates
stop_words.txt            # Global stop words list used across all topics

documents/
  - ccp/                  # Chinese Communist Party topic
    - human/
      - center/
        - wikipedia_ccp.txt
      - american/
        - usa_government_ccp.txt
      - chinese/
        - chinese_government_ccp.txt
    - llm/
      - claude/
        - claude_1_ccp.txt
        - claude_2_ccp.txt
      - deepseek/
        - deepseek_1_ccp.txt
        - deepseek_2_ccp.txt
      - grok/
        - grok_1_ccp.txt
        - grok_2_ccp.txt
    - support/
      - stop_words.txt    # Topic-specific stop words
      - color_mapping.txt # Topic-specific stop words
   
# Or to generalize the format...

  - {topic}/              # e.g., "ccp", "roe_vs_wade"
    - human/              # Human-authored baseline documents
      - {position}/       # e.g., "left", "right", "center"
        - {document}.txt  # Baseline document
        - {document2}.txt  # Baseline document
    - llm/                # LLM-generated documents
      - {llm_provider}/   # e.g., "claude", "grok", "openai"
        - {document}.txt  # LLM output
        - {document2}.txt  # LLM output
    - support/            # Support files
      - stop_words.txt    # Domain-specific stop words
      - color_mapping.txt # Color data for each baseline position
   
  - {roe_vs_wade}/              # e.g., "ccp", "roe_vs_wade", "tariffs"
  - {tariffs}/
```

This modular structure supports multi-topic analysis with minimal code changes - simply switching a single variable (`TOPIC`) in the application file.

## Visualizations (with detailed descriptions)

1. **Similarity Heatmap**
   - Grid visualization with baselines on one axis, LLMs on the other
   - Hierarchical clustering to group similar political tendencies
   - Numeric overlay showing exact similarity scores

2. **Most Similar Baseline Classification**
   - Color-coded boxes showing each LLM's primary political alignment
   - Put the score itself in the box

3. **Political Bias Horizontal Bars**
   - Horizontal stacked bars (one per LLM) showing percentage breakdown of political leanings; sum should add up to 100%
   - Sorted to highlight models with strongest political tendencies

4. **Subplot graph with Per-Model Baselines Comparison**
   - Individual subplots for each LLM showing similarity score to all baselines
   - Deviation from mean highlighted to emphasize distinctive positioning

5. **Common Words Sankey Diagram**
   - Connections between sources and the top 6 common words across all texts
   - Flow width proportional to term frequency in each source

6. **Distinctive Words Sankey Diagram**
   - Maps each source to its 5 most common words
   - Flow width indicating count of the words

7. **Linguistic Radar Plot**
   - Multi-dimensional comparison of writing characteristics on several axes
   - sentence length, word length, lexical diversity, unique words, total word count
   - Scaled axes

8. **PCA Visualization**

## Implementation Details

The system consists of two primary Python files:

1. **CapTuring.py**: Core class implementing the NLP framework

2. **CapTuring_app.py**: Application layer with simple variables to change:
- TOPIC selects which topic/directory (ccp, roe_vs_wade, tariffs, etc)
- GROUPED selects whether .txt files are grouped by LLM/baseline (Claude, OpenAI, left, center, right) or are treated as individual files
- GRAPHS lists each visualization so users can toggle them off or on.

## TO DO LIST

I have presented an overview of how we envision the final product, however the current code is not fully completed. In making future edits, pay attention to the following:

LIST:
- most obviously, I must improve data ingestion.

### Immediate Assignment Requirements
1. **Custom Parser Support**: While the framework has a `parsers` dictionary, we need to improve the ability for users to register their own domain-specific parsers beyond the simple text parser
2. **Two Required Visualizations**: We've implemented the Sankey diagram, but need to fully develop:
   - The subplot visualization (our `visualize_text_features` method needs refinement)
   - A proper comparative visualization that clearly distinguishes between texts

### Additional Planned Features
1. **Advanced Visualizations**:
   - Similarity Heatmap with hierarchical clustering
   - Political bias visualization (horizontal bars, classification boxes)
   - Sentiment trajectory analysis
   - 2D/3D PCA for document positioning

2. **Analysis Enhancements**:
   - Sentiment analysis integration
   - Topic modeling with LDA
   - Certainty language detection
   - More sophisticated group-level analysis beyond simple averaging

3. **Framework Extensibility**:
   - Clearer documentation for extending with custom parsers
   - Standardized interface for adding new analysis metrics
   - More robust state variable structure following assignment guidelines
   - Support for color_mapping.txt and other configuration files

4. **Technical Debt**:
   - Refactor document processing pipeline to support the advanced metrics
   - Enhance error handling for invalid documents or parsing failures
   - Create usage examples for extending the framework
   - Add unit tests for core functionality

I'm excited to begin editing these files!