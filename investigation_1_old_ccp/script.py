import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Define filepath
filepath = 'ccp_documents'

# 1. Load and categorize texts
reference_files = ['wikipedia_ccp.txt', 'chinese_government_ccp.txt', 'american_government_ccp.txt']
llm_files = ['claude_1_ccp.txt', 'claude_2_ccp.txt', 'deepseek_1_ccp.txt', 
             'deepseek_2_ccp.txt', 'grok_1_ccp.txt', 'grok_2_ccp.txt']

# Load reference texts
reference_texts = []
reference_names = []
for file in reference_files:
    with open(os.path.join(filepath, file), 'r', encoding='utf-8') as f:
        reference_texts.append(f.read())
        # Clean up names for display
        name = file.replace('.txt', '').replace('_ccp', '')
        reference_names.append(name)

# Load LLM texts
llm_texts = []
llm_names = []
for file in llm_files:
    with open(os.path.join(filepath, file), 'r', encoding='utf-8') as f:
        llm_texts.append(f.read())
        # Clean up names for display
        name = file.replace('.txt', '').replace('_ccp', '')
        llm_names.append(name)

# 2. Convert all texts to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
all_texts = reference_texts + llm_texts
all_names = reference_names + llm_names
tfidf_matrix = vectorizer.fit_transform(all_texts)

# 3. Calculate similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# 4. Extract similarity scores between LLMs and reference texts
llm_ref_similarity = similarity_matrix[len(reference_texts):, :len(reference_texts)]

# Create a DataFrame for easier manipulation
similarity_df = pd.DataFrame(llm_ref_similarity, 
                            index=llm_names, 
                            columns=reference_names)

# Create output directory if it doesn't exist
os.makedirs('ccp_analysis_output', exist_ok=True)

# 5. VISUALIZATION 1: Heatmap of LLM vs Reference similarities
plt.figure(figsize=(12, 8))
sns.heatmap(similarity_df, annot=True, fmt=".3f", cmap='YlGnBu', 
            vmin=0, vmax=1)
plt.title('Similarity Between LLM Outputs and Reference Texts', fontsize=16)
plt.ylabel('LLM Output', fontsize=12)
plt.xlabel('Reference Source', fontsize=12)
plt.tight_layout()
plt.savefig('ccp_analysis_output/llm_reference_similarity.png')
plt.close()

# 6. VISUALIZATION 2: Bar chart showing which reference each LLM output is most similar to
closest_ref = similarity_df.idxmax(axis=1)
max_similarity = similarity_df.max(axis=1)

# Create a color map based on which reference is closest
color_map = {'wikipedia': 'gray', 'chinese_government': 'red', 'american_government': 'blue'}
bar_colors = [color_map[ref] for ref in closest_ref]

plt.figure(figsize=(12, 7))
bars = plt.bar(llm_names, max_similarity, color=bar_colors)

# Add labels to show which reference is closest
for i, bar in enumerate(bars):
    plt.text(i, bar.get_height() + 0.01, 
             closest_ref[i].replace('_', ' ').title(), 
             ha='center', va='bottom',
             fontweight='bold', rotation=0)

plt.ylim(0, 1)
plt.title('Highest Similarity Score and Closest Reference for Each LLM Output', fontsize=16)
plt.ylabel('Similarity Score', fontsize=12)
plt.xlabel('LLM Output', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ccp_analysis_output/closest_reference.png')
plt.close()

# 7. VISUALIZATION 3: Radar charts for each LLM type
# Group by LLM type and average their similarities
claude_avg = similarity_df.loc[['claude_1', 'claude_2']].mean()
deepseek_avg = similarity_df.loc[['deepseek_1', 'deepseek_2']].mean()
grok_avg = similarity_df.loc[['grok_1', 'grok_2']].mean()

# Create radar chart
fig = plt.figure(figsize=(15, 5))

# Categories (reference texts)
categories = [name.replace('_', ' ').title() for name in reference_names]
N = len(categories)

# Create angle for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Create subplot for Claude
ax1 = fig.add_subplot(131, polar=True)
values = claude_avg.values.tolist()
values += values[:1]  # Close the loop
ax1.plot(angles, values, linewidth=2, linestyle='solid')
ax1.fill(angles, values, alpha=0.25)
ax1.set_thetagrids(np.degrees(angles[:-1]), categories)
ax1.set_ylim(0, 0.7)  # Adjusted for better visualization
ax1.set_title('CLAUDE', fontsize=14)

# Create subplot for Deepseek
ax2 = fig.add_subplot(132, polar=True)
values = deepseek_avg.values.tolist()
values += values[:1]  # Close the loop
ax2.plot(angles, values, linewidth=2, linestyle='solid')
ax2.fill(angles, values, alpha=0.25)
ax2.set_thetagrids(np.degrees(angles[:-1]), categories)
ax2.set_ylim(0, 0.7)  # Adjusted for better visualization
ax2.set_title('DEEPSEEK', fontsize=14)

# Create subplot for grok
ax3 = fig.add_subplot(133, polar=True)
values = grok_avg.values.tolist()
values += values[:1]  # Close the loop
ax3.plot(angles, values, linewidth=2, linestyle='solid')
ax3.fill(angles, values, alpha=0.25)
ax3.set_thetagrids(np.degrees(angles[:-1]), categories)
ax3.set_ylim(0, 0.7)  # Adjusted for better visualization
ax3.set_title('grok', fontsize=14)

plt.tight_layout()
plt.savefig('ccp_analysis_output/llm_radar_charts.png')
plt.close()

# 8. VISUALIZATION 4: Sentiment analysis
# Add sentiment analysis for all texts
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

sentiment_polarity = []
sentiment_subjectivity = []
for text in all_texts:
    polarity, subjectivity = get_sentiment(text)
    sentiment_polarity.append(polarity)
    sentiment_subjectivity.append(subjectivity)

# Create a DataFrame for sentiment
sentiment_df = pd.DataFrame({
    'Text': all_names,
    'Polarity': sentiment_polarity,
    'Subjectivity': sentiment_subjectivity,
    'Type': ['Reference']*len(reference_names) + ['LLM']*len(llm_names),
    'Source': reference_names + llm_names
})

# Plot sentiment
plt.figure(figsize=(12, 8))

# Define source-based colors
source_colors = {
    'wikipedia': 'gray',
    'chinese_government': 'red',
    'american_government': 'blue',
    'claude_1': 'purple', 'claude_2': 'purple',
    'deepseek_1': 'green', 'deepseek_2': 'green',
    'grok_1': 'orange', 'grok_2': 'orange'
}

# Create scatter plot with custom colors
for i, row in sentiment_df.iterrows():
    plt.scatter(row['Polarity'], row['Subjectivity'], 
                color=source_colors[row['Source']], 
                s=150 if row['Type'] == 'Reference' else 80, 
                alpha=0.8,
                edgecolors='black', linewidth=1)
    
    # Add text labels
    plt.annotate(row['Source'].replace('_', ' ').title(), 
                (row['Polarity'], row['Subjectivity']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9)

plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

# Add a custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Wikipedia (Neutral)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Chinese Government (Pro)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='American Government (Anti)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Claude'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Deepseek'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='grok')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.xlabel('Sentiment Polarity (Negative to Positive)', fontsize=12)
plt.ylabel('Subjectivity (Objective to Subjective)', fontsize=12)
plt.title('Sentiment Analysis of CCP Descriptions', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('ccp_analysis_output/sentiment_analysis.png')
plt.close()

# 9. VISUALIZATION 5: Term frequency comparison for politically charged words
# Define politically significant words related to CCP descriptions
political_terms = [
    'communist', 'party', 'government', 'dictatorship', 'democracy', 'freedom', 
    'control', 'power', 'leader', 'leadership', 'authoritarian', 'regime', 
    'socialist', 'marxist', 'state', 'people', 'rights', 'human', 'policy', 
    'economy', 'military', 'political', 'chairman', 'ideology', 'revolution',
    'censorship', 'oppression', 'propaganda', 'liberalization', 'market'
]

# Function to count term frequency
def count_terms(text, terms):
    text = text.lower()
    counts = {}
    for term in terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        counts[term] = len(re.findall(pattern, text))
    return counts

# Count terms in all texts
term_counts = []
for text in all_texts:
    term_counts.append(count_terms(text, political_terms))

# Create a DataFrame
term_df = pd.DataFrame(term_counts, index=all_names)

# Normalize counts (convert to percentages of max for each term)
term_df_norm = term_df.apply(lambda x: x / x.max() if x.max() > 0 else x, axis=0)

# Filter to keep only terms that appear at least once
term_df_norm = term_df_norm.loc[:, (term_df_norm > 0).any(axis=0)]

# Sort columns by total usage
term_df_norm = term_df_norm[term_df_norm.sum().sort_values(ascending=False).index]

# Keep only top 15 most used terms for readability
top_terms = term_df_norm.columns[:15]
term_df_filtered = term_df_norm[top_terms]

# VISUALIZATION 5: Heatmap of political terms
plt.figure(figsize=(16, 10))
sns.heatmap(term_df_filtered, cmap='YlGnBu', annot=True, fmt=".2f", cbar_kws={'label': 'Normalized Frequency'})
plt.title('Political Term Usage in CCP Descriptions (Normalized)', fontsize=16)
plt.ylabel('Source', fontsize=12)
plt.xlabel('Political Term', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('ccp_analysis_output/political_terms_heatmap.png')
plt.close()

# 10. VISUALIZATION 6: PCA clustering to visualize text relationships
# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Create a plot
plt.figure(figsize=(12, 8))

# Plot reference texts with specific colors
ref_colors = ['gray', 'red', 'blue']
for i, name in enumerate(reference_names):
    plt.scatter(pca_result[i, 0], pca_result[i, 1], 
                c=ref_colors[i], s=150, alpha=0.8, 
                edgecolors='black', linewidth=1)
    plt.annotate(name.replace('_', ' ').title(), 
                (pca_result[i, 0], pca_result[i, 1]),
                fontsize=10, fontweight='bold')

# Plot LLM texts with model-specific colors
llm_colors = {
    'claude': 'purple',
    'deepseek': 'green',
    'grok': 'orange'
}

for i, name in enumerate(llm_names):
    model_type = name.split('_')[0]
    plt.scatter(pca_result[i+len(reference_names), 0], 
                pca_result[i+len(reference_names), 1], 
                c=llm_colors[model_type], s=100, alpha=0.7,
                edgecolors='black', linewidth=1)
    plt.annotate(name.replace('_', ' ').title(), 
                (pca_result[i+len(reference_names), 0], 
                 pca_result[i+len(reference_names), 1]),
                fontsize=9)

# Add a custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Wikipedia (Neutral)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Chinese Government (Pro)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='American Government (Anti)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Claude'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Deepseek'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='grok')
]
plt.legend(handles=legend_elements, loc='best')

plt.title('PCA Clustering of CCP Descriptions', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('ccp_analysis_output/text_clustering.png')
plt.close()

# 11. VISUALIZATION 7: Calculate and display bias scores
# For each LLM output, compute a "bias score" that shows relative similarity to each reference
bias_scores = pd.DataFrame(index=llm_names, columns=[
    'Wikipedia (Neutral)', 
    'Chinese Govt (Pro)', 
    'American Govt (Anti)'
])

for llm in llm_names:
    # Get the similarities to each reference
    wiki_sim = similarity_df.loc[llm, 'wikipedia']
    chinese_sim = similarity_df.loc[llm, 'chinese_government']
    american_sim = similarity_df.loc[llm, 'american_government']
    
    # Calculate total similarity
    total_sim = wiki_sim + chinese_sim + american_sim
    
    # Calculate bias as percentage of total similarity
    bias_scores.loc[llm, 'Wikipedia (Neutral)'] = wiki_sim / total_sim
    bias_scores.loc[llm, 'Chinese Govt (Pro)'] = chinese_sim / total_sim
    bias_scores.loc[llm, 'American Govt (Anti)'] = american_sim / total_sim

# Convert to percentages for better readability
bias_scores = bias_scores * 100

# Plot stacked bar chart of bias percentages
plt.figure(figsize=(14, 8))
bias_scores.plot(kind='bar', stacked=True, 
                 color=['gray', 'red', 'blue'], 
                 figsize=(14, 8))
plt.title('Relative Bias of LLM Outputs - Percentage of Total Similarity', fontsize=16)
plt.xlabel('LLM Output', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Reference Source')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('ccp_analysis_output/bias_percentage.png')
plt.close()

# 12. Generate a summary report
with open('ccp_analysis_output/analysis_summary.txt', 'w') as f:
    f.write("CCP DESCRIPTION BIAS ANALYSIS SUMMARY\n")
    f.write("====================================\n\n")
    
    f.write("CLOSEST REFERENCE SOURCE FOR EACH LLM:\n")
    for llm in llm_names:
        closest = similarity_df.loc[llm].idxmax()
        score = similarity_df.loc[llm, closest]
        f.write(f"- {llm.replace('_', ' ').title()}: {closest.replace('_', ' ').title()} (similarity: {score:.3f})\n")
    
    f.write("\nBIAS DISTRIBUTION (PERCENTAGE):\n")
    for llm in llm_names:
        f.write(f"- {llm.replace('_', ' ').title()}:\n")
        for ref in bias_scores.columns:
            f.write(f"  - {ref}: {bias_scores.loc[llm, ref]:.1f}%\n")
    
    f.write("\nSENTIMENT ANALYSIS:\n")
    for i, row in sentiment_df.iterrows():
        f.write(f"- {row['Source'].replace('_', ' ').title()}: ")
        f.write(f"Polarity = {row['Polarity']:.3f} (")
        if row['Polarity'] > 0.05:
            f.write("Positive")
        elif row['Polarity'] < -0.05:
            f.write("Negative")
        else:
            f.write("Neutral")
        f.write(f"), Subjectivity = {row['Subjectivity']:.3f}\n")

print("Analysis complete! Results saved to the 'ccp_analysis_output' directory.")