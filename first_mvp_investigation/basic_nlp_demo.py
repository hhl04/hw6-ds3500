import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# 1. Load all texts with your specific filepath
filepath = '/Users/seanblundin/Documents/courses/ds3500_not_fucked_by_github/hw6-ds3500/documents'
texts = []
filenames = []
for file in os.listdir(filepath):
    if file.endswith('.txt'):
        with open(os.path.join(filepath, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            filenames.append(file)

# 2. Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Get simplified filenames (without .txt) for better visualization
simple_names = [fname.replace('.txt', '') for fname in filenames]

# 3. Calculate similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# 4. VISUALIZATION 1: Similarity Heatmap with actual filenames
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=simple_names, 
            yticklabels=simple_names)
plt.title('Similarity Between Different LLM Responses')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('similarity_heatmap.png')
plt.close()

# 5. VISUALIZATION 2: Hierarchical Clustering Dendrogram
plt.figure(figsize=(12, 8))
linked = linkage(similarity_matrix, 'ward')
dendrogram(linked, labels=simple_names, orientation='top', 
           leaf_font_size=9)
plt.title('Hierarchical Clustering of LLM Responses')
plt.xlabel('LLM Response')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('dendrogram.png')
plt.close()

# 6. VISUALIZATION 3: PCA to visualize in 2D space
pca = PCA(n_components=2)
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

plt.figure(figsize=(10, 8))
plt.scatter(tfidf_pca[:, 0], tfidf_pca[:, 1], s=100)
for i, txt in enumerate(simple_names):
    plt.annotate(txt, (tfidf_pca[i, 0], tfidf_pca[i, 1]), fontsize=9)
plt.title('PCA of LLM Responses')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('pca_plot.png')
plt.close()

# 7. VISUALIZATION 4: Word count comparison
word_counts = []
unique_words = []

for text in texts:
    words = text.lower().split()
    word_counts.append(len(words))
    unique_words.append(len(set(words)))

# Create a DataFrame and plot
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(filenames))

plt.bar(index, word_counts, bar_width, label='Total Words')
plt.bar(index + bar_width, unique_words, bar_width, label='Unique Words')

plt.xlabel('LLM Response')
plt.ylabel('Word Count')
plt.title('Word Usage Comparison Across LLMs')
plt.xticks(index + bar_width/2, simple_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('word_counts.png')
plt.close()

# 8. VISUALIZATION 5: Topic focus (top words from each response)
plt.figure(figsize=(12, 10))
vectorizer = CountVectorizer(stop_words='english', max_features=30)
X = vectorizer.fit_transform(texts)
words = vectorizer.get_feature_names_out()
word_counts = X.toarray()

# Create a heatmap of top words
sns.heatmap(word_counts, cmap='YlGnBu',
            xticklabels=words, 
            yticklabels=simple_names)
plt.title('Top Word Usage Across LLM Responses')
plt.tight_layout()
plt.savefig('topic_heatmap.png')
plt.close()

# 9. VISUALIZATION 6: Word Clouds (one for each text)
os.makedirs('wordclouds', exist_ok=True)
for i, text in enumerate(texts):
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud: {simple_names[i]}')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'wordclouds/wordcloud_{simple_names[i]}.png')
    plt.close()

print("Analysis complete. All visualizations have been saved.")