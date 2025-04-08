# CapTuring: Analyzing Political Bias in LLMs

CapTuring is an NLP framework designed to quantify and visualize political bias in large language model outputs through comparative text analysis. We use methods like cosine similarity on TF-IDF representations to analyze LLM-generated content along a political spectrum defined by human-authored baseline documents.

For example, we first choose a topic such as the CCP, Roe vs. Wade, or Trump's Tariffs. Then, we select human-written documents (usually one each for "left" like CNN, "center" like Wikipedia or AP, and "right" like Fox News) as our baseline documents for comparison. We then ask different LLMs such as Claude, OpenAI, Google Gemini, or Deepseek to write an article (usually two articles per LLM) about the topic.

Then, we're able to compare these LLM outputs to the human-written biased documents to examine their closeness to each, and therefore measure each LLM's "bias".

## Research Question

The core research question is: **To what extent do different large language models exhibit political biases in their outputs, and how do these biases compare across different LLMs and topics?**

## Methodology

### Technical Details

1. **Vector Representation**: Documents are transformed into vector space using TF-IDF vectorization
2. **Similarity Metrics**: Cosine similarity is used to measure document relatedness
3. **Baseline Anchoring**: Human-authored documents represent political positions (left, right, center, etc.)
4. **Multi-dimensional Analysis**: Documents are positioned in n-dimensional political space where n = number of baseline positions
5. **Group-level Aggregation**: Multiple documents from the same LLM are treated as a cohesive group for higher-level analysis

### Data Structure

```
documents/
  - {topic}/              # e.g., "ccp", "roe_vs_wade"
    - human/              # Human-authored baseline documents
      - {position}/       # e.g., "left", "right", "center"
        - {document}.txt  # Baseline document
    - llm/                # LLM-generated documents
      - {llm_provider}/   # e.g., "claude", "grok", "openai"
        - {document}.txt  # LLM outputs
    - support/            # Support files
      - stop_words.txt    # Domain-specific stop words
      - color_mapping.txt # Color data for each baseline position
```

For example with CCP data:

documents/
  - ccp/
    - human/
      - center/
        - wikipedia_ccp.txt
      - left/
        - american_government_ccp.txt
      - right/
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
      - stop_words_ccp.txt
      - color_mapping.txt
  - {topic 2}/              # e.g., "ccp", "roe_vs_wade", "tariffs"
  - {topic 3}/
  - {topic 4}/
  - {topic 5}/

This modular structure supports multi-topic analysis with minimal code changes - simply switching a single variable (`TOPIC`) in the application file.

## Analysis Pipeline (approximate)

### 1. Data Processing
   - Document collection and preprocessing
   - Stop word filtering
   - TF-IDF vectorization

### 2. Similarity Calculation
   - Document-to-document similarity matrix
   - Political baseline mapping
   - Group-level aggregation

### 3. Visualization & Analysis
   - Political bias distribution charts
   - Position mapping in political space
   - Distinctive term identification

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

8. **Linguistic Complexity Comparison**
   - Because some political positions might systematically use simpler/complex language

9. **Certainty Language Analysis**
   - Because some political positions or LLM responses might use more uncertain language.

10. **Sentiment Trajectory**
    - Line charts showing emotional tone changes throughout the progression of each txt document
    - Overlaid comparison of sentiment patterns by political position

11. **Topic Distribution Comparison**
    - Automatically extracted topics using LDA with top terms labeled
    - Topic prevalence normalized as percentage of document content

12. **2D PCA Visualization**

13. **3D PCA Visualization**

## Implementation Details

The system consists of two primary Python files:

1. **CapTuring.py**: Core class implementing the NLP framework
   - Document loading and processing
   - Vector representation and similarity calculations
   - Group-level analysis functions
   - Visualization methods

2. **CapTuring_app.py**: Application layer
   - Topic/directory selector
   - console updates