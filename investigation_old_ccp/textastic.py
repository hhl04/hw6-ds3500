"""
File: textastic.py

Description: A reusable, extensible framework
for comparitive text analysis designed to work
with any arbitrary collection of related documents.
"""

from collections import Counter, defaultdict
import random as rnd
import matplotlib.pyplot as plt
import math
import re
import numpy as np
from matplotlib.lines import Line2D


class Textastic:

    def __init__(self):
        """ Constructor """
        self.data = defaultdict(dict)
        self.documents = {}  # Store raw text
        self.document_groups = {}  # For grouping documents (e.g., reference vs LLM)
        self.stop_words = set()  # Set of stop words to filter

    def load_stop_words(self, filename):
        """Load stop words from a file"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                stop_words = file.read().split()
                self.stop_words = set(word.lower() for word in stop_words)
            print(f"Loaded {len(self.stop_words)} stop words from {filename}")
        except Exception as e:
            print(f"Error loading stop words from {filename}: {e}")

    def simple_text_parser(self, filename):
        """ For processing simple, unformatted text documents """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Clean text (remove punctuation and convert to lowercase)
                clean_text = re.sub(r'[^\w\s]', '', text.lower())
                words = clean_text.split()
                
                # Filter out stop words
                if self.stop_words:
                    filtered_words = [word for word in words if word not in self.stop_words]
                else:
                    filtered_words = words
                
                # Create results
                results = {
                    'wordcount': Counter(filtered_words),
                    'numwords': len(filtered_words),
                    'text': text,
                    'clean_text': ' '.join(filtered_words),
                    'raw_wordcount': Counter(words),  # Keep original word count too
                    'raw_numwords': len(words)
                }
                
                return results
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return {
                'wordcount': Counter(),
                'numwords': 0,
                'text': "",
                'clean_text': "",
                'raw_wordcount': Counter(),
                'raw_numwords': 0
            }

    def load_text(self, filename, label=None, parser=None, group=None):
        """ Register a document with the framework and
        store data extracted from the document to be used
        later in visualizations """

        results = self.simple_text_parser(filename)  # default
        if parser is not None:
            # If custom parser is provided, pass stop words to it
            if hasattr(parser, '__code__') and 'stop_words' in parser.__code__.co_varnames:
                results = parser(filename, stop_words=self.stop_words)
            else:
                results = parser(filename)

        if label is None:
            label = filename

        # Store raw document
        self.documents[label] = results.get('text', '')
        
        # Add to group if specified
        if group:
            if group not in self.document_groups:
                self.document_groups[group] = []
            self.document_groups[group].append(label)

        # Store results in the data dictionary
        for k, v in results.items():
            self.data[k][label] = v

    def compare_num_words(self):
        """ A very simplistic visualization that creates a bar
        chart comparing the number of words in each file. """

        num_words = self.data['numwords']
        plt.figure(figsize=(10, 6))
        plt.bar(num_words.keys(), num_words.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('Document Word Count Comparison (After Stop Word Removal)')
        plt.ylabel('Number of Words')
        plt.tight_layout()
        plt.show()
        
    def analyze_stop_word_impact(self):
        """Visualize the impact of stop word removal on word counts"""
        # Compare raw counts with filtered counts
        if 'raw_numwords' not in self.data:
            print("Raw word counts not available. Can't analyze stop word impact.")
            return
            
        # Get word counts before and after filtering
        raw_counts = {doc: self.data['raw_numwords'][doc] for doc in self.data['raw_numwords']}
        filtered_counts = {doc: self.data['numwords'][doc] for doc in self.data['numwords']}
        
        # Calculate percentage of stop words
        stop_word_percentage = {}
        for doc in raw_counts:
            if raw_counts[doc] > 0:
                stop_word_percentage[doc] = ((raw_counts[doc] - filtered_counts[doc]) / raw_counts[doc]) * 100
            else:
                stop_word_percentage[doc] = 0
        
        # Visualize
        plt.figure(figsize=(12, 8))
        
        # Plot raw vs filtered word counts
        x = range(len(raw_counts))
        width = 0.35
        
        plt.subplot(2, 1, 1)
        plt.bar(x, list(raw_counts.values()), width, label='With Stop Words')
        plt.bar([i+width for i in x], list(filtered_counts.values()), width, label='Stop Words Removed')
        plt.xticks([i+width/2 for i in x], list(raw_counts.keys()), rotation=45, ha='right')
        plt.title('Word Count Before and After Stop Word Removal')
        plt.ylabel('Word Count')
        plt.legend()
        plt.tight_layout()
        
        # Plot percentage of stop words
        plt.subplot(2, 1, 2)
        plt.bar(stop_word_percentage.keys(), stop_word_percentage.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('Percentage of Stop Words in Each Document')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def get_all_words(self):
        """ Get all unique words across all documents """
        all_words = set()
        for doc, wordcount in self.data['wordcount'].items():
            all_words.update(wordcount.keys())
        return all_words

    def create_document_term_matrix(self):
        """ Create a document-term matrix (DTM) from loaded documents """
        # Get all unique words
        all_words = sorted(list(self.get_all_words()))
        
        # Create the DTM
        dtm = {}
        for doc, wordcount in self.data['wordcount'].items():
            dtm[doc] = [wordcount.get(word, 0) for word in all_words]
            
        return dtm, all_words

    def compute_tf(self, term_count, total_terms):
        """ Compute Term Frequency (TF) """
        return term_count / total_terms if total_terms > 0 else 0

    def compute_idf(self, term, documents):
        """ Compute Inverse Document Frequency (IDF) """
        # Count how many documents contain the term
        doc_count = sum(1 for doc, counts in documents.items() if term in counts)
        
        # Avoid division by zero
        if doc_count == 0:
            return 0
            
        # Calculate IDF
        return math.log(len(documents) / doc_count)

    def create_tfidf_matrix(self):
        """ Create a TF-IDF matrix from loaded documents """
        # Get all unique words
        all_words = sorted(list(self.get_all_words()))
        
        # Initialize the TF-IDF matrix
        tfidf_matrix = {}
        
        # Calculate TF-IDF for each document and term
        for doc, wordcount in self.data['wordcount'].items():
            total_terms = self.data['numwords'][doc]
            tfidf_matrix[doc] = []
            
            for word in all_words:
                # Calculate term frequency (TF)
                tf = self.compute_tf(wordcount.get(word, 0), total_terms)
                
                # Calculate inverse document frequency (IDF)
                idf = self.compute_idf(word, self.data['wordcount'])
                
                # Calculate TF-IDF
                tfidf = tf * idf
                tfidf_matrix[doc].append(tfidf)
        
        return tfidf_matrix, all_words

    def compute_cosine_similarity(self, vec1, vec2):
        """ Compute cosine similarity between two vectors """
        # Convert lists to numpy arrays for easier calculation
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes
        mag1 = np.sqrt(np.sum(np.square(vec1)))
        mag2 = np.sqrt(np.sum(np.square(vec2)))
        
        # Avoid division by zero
        if mag1 * mag2 == 0:
            return 0
            
        # Calculate cosine similarity
        return dot_product / (mag1 * mag2)

    def compute_all_similarities(self):
        """ Compute similarities between all document pairs """
        # Create TF-IDF matrix
        tfidf_matrix, _ = self.create_tfidf_matrix()
        
        # Compute similarity for each document pair
        similarities = {}
        for doc1 in tfidf_matrix:
            similarities[doc1] = {}
            for doc2 in tfidf_matrix:
                similarities[doc1][doc2] = self.compute_cosine_similarity(
                    tfidf_matrix[doc1], tfidf_matrix[doc2])
        
        return similarities

    def compute_group_similarities(self, group1, group2):
        """ Compute similarities between documents of two groups """
        # Validate groups
        if not (group1 in self.document_groups and group2 in self.document_groups):
            print(f"Group(s) not found: {group1}, {group2}")
            return {}
            
        # Get documents in each group
        docs1 = self.document_groups[group1]
        docs2 = self.document_groups[group2]
        
        # Compute all similarities
        all_similarities = self.compute_all_similarities()
        
        # Extract similarities between groups
        group_similarities = {}
        for doc1 in docs1:
            group_similarities[doc1] = {doc2: all_similarities[doc1][doc2] for doc2 in docs2}
            
        return group_similarities

    def visualize_group_similarities(self, group1, group2, title=None):
        """ Visualize similarities between documents of two groups """
        # Compute similarities
        similarities = self.compute_group_similarities(group1, group2)
        
        if not similarities:
            print("No similarities found.")
            return
            
        # Prepare data for visualization
        docs1 = list(similarities.keys())
        docs2 = list(similarities[docs1[0]].keys())
        
        # Create a matrix of similarities
        sim_matrix = np.zeros((len(docs1), len(docs2)))
        for i, doc1 in enumerate(docs1):
            for j, doc2 in enumerate(docs2):
                sim_matrix[i, j] = similarities[doc1][doc2]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(sim_matrix, cmap='YlGnBu', aspect='auto')
        plt.colorbar(label='Similarity Score')
        
        # Set labels
        plt.xticks(range(len(docs2)), docs2, rotation=45, ha='right')
        plt.yticks(range(len(docs1)), docs1)
        
        plt.title(title or f'Similarity between {group1} and {group2}')
        plt.tight_layout()
        plt.show()
        
        return sim_matrix

    def find_closest_reference(self, llm_doc, reference_docs):
        """Find which reference document an LLM doc is most similar to"""
        # Compute all similarities
        all_similarities = self.compute_all_similarities()
        
        # Find the most similar reference
        max_sim = -1
        closest_ref = None
        
        for ref_doc in reference_docs:
            sim = all_similarities[llm_doc][ref_doc]
            if sim > max_sim:
                max_sim = sim
                closest_ref = ref_doc
                
        return closest_ref, max_sim

    def calculate_bias_scores(self, llm_docs, reference_docs):
        """Calculate bias scores for LLM documents relative to reference documents"""
        # Compute all similarities
        all_similarities = self.compute_all_similarities()
        
        # Calculate bias scores
        bias_scores = {}
        for llm_doc in llm_docs:
            # Get similarities to all references
            ref_sims = {ref: all_similarities[llm_doc][ref] for ref in reference_docs}
            
            # Calculate total similarity
            total_sim = sum(ref_sims.values())
            if total_sim == 0:
                continue
                
            # Calculate relative bias (as percentage of total similarity)
            bias_scores[llm_doc] = {ref: (sim / total_sim) * 100 for ref, sim in ref_sims.items()}
            
        return bias_scores

    def visualize_bias_distribution(self, llm_group, reference_group, title=None):
        """Visualize bias distribution of LLM docs towards reference docs"""
        # Get docs in each group
        llm_docs = self.document_groups.get(llm_group, [])
        ref_docs = self.document_groups.get(reference_group, [])
        
        if not llm_docs or not ref_docs:
            print("Group(s) not found or empty.")
            return
            
        # Calculate bias scores
        bias_scores = self.calculate_bias_scores(llm_docs, ref_docs)
        
        if not bias_scores:
            print("No bias scores calculated.")
            return
            
        # Prepare data for visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Position of each bar on x-axis
        x = np.arange(len(llm_docs))
        width = 0.8 / len(ref_docs)  # Width of each bar
        
        # Plot bars for each reference
        for i, ref in enumerate(ref_docs):
            values = [bias_scores[llm].get(ref, 0) for llm in llm_docs]
            ax.bar(x + i * width - width * (len(ref_docs) - 1) / 2, 
                   values, width, label=ref)
        
        # Add labels and title
        ax.set_ylabel('Bias Score (%)')
        ax.set_xlabel('LLM Document')
        ax.set_title(title or f'Bias Distribution of {llm_group} towards {reference_group}')
        ax.set_xticks(x)
        ax.set_xticklabels(llm_docs, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_closest_references(self, llm_group, reference_group, title=None):
        """Plot which reference each LLM doc is most similar to"""
        # Get docs in each group
        llm_docs = self.document_groups.get(llm_group, [])
        ref_docs = self.document_groups.get(reference_group, [])
        
        if not llm_docs or not ref_docs:
            print("Group(s) not found or empty.")
            return
            
        # For each LLM doc, find the closest reference
        closest_refs = {}
        for llm_doc in llm_docs:
            closest_ref, max_sim = self.find_closest_reference(llm_doc, ref_docs)
            closest_refs[llm_doc] = (closest_ref, max_sim)
            
        # Prepare for visualization
        plt.figure(figsize=(12, 8))
        
        # Assign colors to references
        ref_colors = {}
        colors = ['gray', 'red', 'blue', 'green', 'purple', 'orange']
        for i, ref in enumerate(ref_docs):
            ref_colors[ref] = colors[i % len(colors)]
            
        # Create bars with colors based on closest reference
        llms = list(closest_refs.keys())
        sims = [closest_refs[llm][1] for llm in llms]
        bar_colors = [ref_colors[closest_refs[llm][0]] for llm in llms]
        
        bars = plt.bar(llms, sims, color=bar_colors)
        
        # Add text on top of each bar showing closest reference
        for i, bar in enumerate(bars):
            plt.text(i, bar.get_height() + 0.01, 
                     closest_refs[llms[i]][0],
                     ha='center', va='bottom',
                     fontsize=8, rotation=45)
        
        # Add legend for references
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=10, label=ref) 
                          for ref, color in ref_colors.items()]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.title(title or 'Closest Reference for Each LLM Output')
        plt.ylabel('Similarity Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def analyze_common_words(self, n=20):
        """Analyze most common words across all documents"""
        # Get all word counts
        all_counts = {}
        for doc, wordcount in self.data['wordcount'].items():
            all_counts[doc] = dict(wordcount.most_common(n))
            
        # Find common words across all documents
        common_words = set()
        for counts in all_counts.values():
            common_words.update(counts.keys())
            
        # Keep only the top n words
        all_words_combined = Counter()
        for counts in self.data['wordcount'].values():
            all_words_combined.update(counts)
            
        top_words = [word for word, _ in all_words_combined.most_common(n)]
        
        # Create a matrix for visualization
        word_matrix = np.zeros((len(all_counts), len(top_words)))
        
        # Fill the matrix
        for i, (doc, counts) in enumerate(all_counts.items()):
            for j, word in enumerate(top_words):
                # Get the count and normalize by total words in document
                count = counts.get(word, 0)
                word_matrix[i, j] = count / self.data['numwords'][doc] if self.data['numwords'][doc] > 0 else 0
                
        # Visualize as heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(word_matrix, aspect='auto', cmap='YlGnBu')
        plt.colorbar(label='Normalized Frequency')
        
        # Set labels
        plt.xticks(range(len(top_words)), top_words, rotation=45, ha='right')
        plt.yticks(range(len(all_counts)), all_counts.keys())
        
        plt.title(f'Top {n} Words Across Documents (Normalized)')
        plt.tight_layout()
        plt.show()
        
        return word_matrix, top_words

    def analyze_political_terms(self, terms, title=None):
        """Analyze frequency of specific political terms"""
        # Check for empty terms list
        if not terms:
            print("No terms provided.")
            return
            
        # Count term occurrences in each document
        term_counts = {}
        for doc, text in self.data['clean_text'].items():
            term_counts[doc] = {}
            for term in terms:
                # Count occurrences (whole word only)
                pattern = r'\b' + re.escape(term) + r'\b'
                count = len(re.findall(pattern, text))
                term_counts[doc][term] = count
                
        # Normalize counts
        norm_counts = {}
        for doc, counts in term_counts.items():
            total_words = self.data['numwords'][doc]
            if total_words > 0:
                norm_counts[doc] = {term: count/total_words for term, count in counts.items()}
            else:
                norm_counts[doc] = {term: 0 for term in terms}
                
        # Create a matrix for visualization
        docs = list(norm_counts.keys())
        term_matrix = np.zeros((len(docs), len(terms)))
        
        # Fill the matrix
        for i, doc in enumerate(docs):
            for j, term in enumerate(terms):
                term_matrix[i, j] = norm_counts[doc][term]
                
        # Visualize as heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(term_matrix, aspect='auto', cmap='YlGnBu')
        plt.colorbar(label='Normalized Frequency')
        
        # Set labels
        plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
        plt.yticks(range(len(docs)), docs)
        
        plt.title(title or 'Political Term Usage Across Documents (Normalized)')
        plt.tight_layout()
        plt.show()
        
        return term_matrix, terms