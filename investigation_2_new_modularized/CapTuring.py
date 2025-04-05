"""
File: CapTuring.py

Description: A reusable, extensible framework for comparative text analysis
designed to analyze and visualize relationships between documents, with a
focus on political spectrum positioning and document similarity.
"""

from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CapTuring:

    def __init__(self):
        """ Constructor """
        self.data = defaultdict(dict)
        self.documents = {}  # Store raw text of documents
        self.baselines = {}  # Store baseline documents
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def simple_text_parser(self, filename):
        """ For processing simple, unformatted text documents """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
                
            words = re.findall(r'\b\w+\b', text.lower())
            word_count = Counter(words)
            
            results = {
                'text': text,
                'wordcount': word_count,
                'numwords': len(words),
                'unique_words': len(word_count)
            }
            
            print(f"Parsed: {filename}: {len(words)} words, {len(word_count)} unique words")
            return results
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return {'text': '', 'wordcount': Counter(), 'numwords': 0, 'unique_words': 0}

    def load_document(self, filename, label=None, parser=None, is_baseline=False, baseline_type=None):
        """ Register a document with the framework and
        store data extracted from the document to be used
        later in visualizations """

        results = self.simple_text_parser(filename)  # default
        if parser is not None:
            results = parser(filename)

        if label is None:
            label = filename

        # Store the document and its metadata
        self.documents[label] = results['text']
        
        # Store all results in the data dictionary
        for k, v in results.items():
            self.data[k][label] = v
            
        # Register as baseline if specified
        if is_baseline and baseline_type is not None:
            self.baselines[baseline_type] = label
            print(f"Registered '{label}' as {baseline_type} baseline")

        # Invalidate existing matrices since we added a new document
        self._invalidate_matrices()
        
        return label  # Return the label for reference

    def _invalidate_matrices(self):
        """Reset computed matrices when documents change"""
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def _ensure_vectorized(self):
        """Ensure TF-IDF matrix is computed"""
        if self.tfidf_matrix is None and self.documents:
            # Get document labels and texts in same order
            labels = list(self.documents.keys())
            texts = [self.documents[label] for label in labels]
            
            # Calculate TF-IDF
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            print(f"Vectorized {len(labels)} documents with {len(self.vectorizer.get_feature_names_out())} features")
            return labels
        return list(self.documents.keys())

    def calculate_similarity_matrix(self):
        """Calculate cosine similarity between all documents"""
        if not self.documents:
            print("No documents loaded")
            return None, []
            
        labels = self._ensure_vectorized()
        
        if self.similarity_matrix is None:
            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            print(f"Calculated similarity matrix of shape {self.similarity_matrix.shape}")
        
        return self.similarity_matrix, labels

    def get_document_similarity(self, doc1_label, doc2_label):
        """Get cosine similarity between two specific documents"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        
        if sim_matrix is None:
            return None
            
        try:
            idx1 = labels.index(doc1_label)
            idx2 = labels.index(doc2_label)
            similarity = sim_matrix[idx1, idx2]
            print(f"Similarity between '{doc1_label}' and '{doc2_label}': {similarity:.4f}")
            return similarity
        except ValueError:
            print(f"Document labels not found. Available labels: {labels}")
            return None

    def calculate_political_spectrum(self):
        """Calculate where each document falls on the political spectrum"""
        if len(self.baselines) < 2:
            print("Need at least two baseline documents (e.g., 'conservative' and 'progressive')")
            return {}
            
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            return {}
            
        spectrum_positions = {}
        
        # Get the indices of baseline documents
        baseline_indices = {btype: labels.index(label) for btype, label in self.baselines.items()}
        
        # For each document, calculate its relative position
        for i, label in enumerate(labels):
            positions = {}
            for btype, idx in baseline_indices.items():
                positions[btype] = sim_matrix[i, idx]
            
            spectrum_positions[label] = positions
            
        return spectrum_positions

    def visualize_similarity_heatmap(self, figsize=(10, 8)):
        """Visualize document similarity as a heatmap"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            return
            
        plt.figure(figsize=figsize)
        plt.imshow(sim_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Cosine Similarity')
        
        # Add labels
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        
        plt.title('Document Similarity Matrix')
        plt.tight_layout()
        plt.show()

    def visualize_political_spectrum(self, figsize=(12, 6), primary_axes=None):
        """Visualize documents on a political spectrum
        
        Args:
            figsize (tuple): Figure size
            primary_axes (list): List of two baseline types to use for x and y axes.
                                If None, the first two baselines are used.
        """
        if len(self.baselines) < 2:
            print("Need at least two baseline documents")
            return
            
        # Get baseline types
        baseline_types = list(self.baselines.keys())
        
        # For 2D visualization, we need to select which baselines to use for axes
        if primary_axes is None:
            # Default to first two baselines for axes
            primary_axes = baseline_types[:2]
        elif not all(axis in baseline_types for axis in primary_axes):
            print(f"Specified axes {primary_axes} not found in baselines {baseline_types}")
            primary_axes = baseline_types[:2]
            
        spectrum_positions = self.calculate_political_spectrum()
        
        # Create the main 2D plot with primary axes
        self._create_2d_political_plot(spectrum_positions, primary_axes, figsize)
        
        # If we have more than 2 baselines, create additional visualizations
        if len(baseline_types) > 2:
            self._create_radar_plot(spectrum_positions, baseline_types)
            self._create_multi_baseline_heatmap(spectrum_positions, baseline_types)
    
    def _create_2d_political_plot(self, spectrum_positions, primary_axes, figsize=(12, 6)):
        """Create a 2D scatter plot for two selected baseline types"""
        # Create position values for plotting
        x_values = []
        y_values = []
        annotations = []
        
        for label, positions in spectrum_positions.items():
            # Use the similarity to each baseline as x and y coordinates
            x = positions.get(primary_axes[0], 0)
            y = positions.get(primary_axes[1], 0)
            
            x_values.append(x)
            y_values.append(y)
            annotations.append(label)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Scatter plot for document positions
        plt.scatter(x_values, y_values, c='blue', alpha=0.6)
        
        # Add labels for each point
        for i, label in enumerate(annotations):
            is_baseline = label in self.baselines.values()
            weight = 'bold' if is_baseline else 'normal'
            plt.annotate(label, (x_values[i], y_values[i]), 
                         fontweight=weight,
                         xytext=(5, 5), textcoords='offset points')
        
        # Add axis labels
        plt.xlabel(f'Similarity to {primary_axes[0]}')
        plt.ylabel(f'Similarity to {primary_axes[1]}')
        
        plt.title('Document Positioning on Political Spectrum (2D)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def _create_radar_plot(self, spectrum_positions, baseline_types, figsize=(10, 8)):
        """Create a radar plot to visualize multi-dimensional baseline similarities"""
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.transforms import Affine2D
        
        # Number of variables
        N = len(baseline_types)
        
        # We need at least 3 dimensions for a radar plot
        if N < 3:
            return
            
        # Select non-baseline documents to display
        non_baseline_docs = [label for label in spectrum_positions.keys() 
                           if label not in self.baselines.values()]
        
        if not non_baseline_docs:
            return
            
        # Set up the angles for each baseline (evenly spaced)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set up the figure
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)
        
        # Draw baseline labels
        plt.xticks(angles[:-1], baseline_types, color='black', size=8)
        
        # Draw y-axis labels (similarity values)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        # Plot each document
        for i, doc in enumerate(non_baseline_docs):
            values = [spectrum_positions[doc].get(baseline, 0) for baseline in baseline_types]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=doc)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Multi-dimensional Political Spectrum Analysis (Radar)')
        plt.show()
    
    def _create_multi_baseline_heatmap(self, spectrum_positions, baseline_types, figsize=(12, 8)):
        """Create a heatmap of similarities to all baselines"""
        # Extract data for the heatmap
        doc_labels = [label for label in spectrum_positions.keys()]
        
        # Create a matrix of similarities
        heatmap_data = np.zeros((len(doc_labels), len(baseline_types)))
        
        for i, doc in enumerate(doc_labels):
            for j, baseline in enumerate(baseline_types):
                heatmap_data[i, j] = spectrum_positions[doc].get(baseline, 0)
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Similarity Score')
        
        # Add labels
        plt.yticks(range(len(doc_labels)), doc_labels)
        plt.xticks(range(len(baseline_types)), baseline_types, rotation=45, ha='right')
        
        # Add title and labels
        plt.title('Document Similarity to Political Baselines')
        plt.tight_layout()
        plt.show()

    def compare_word_frequencies(self, n=20, figsize=(15, 8)):
        """Compare most common words across documents"""
        if not self.data['wordcount']:
            print("No word count data available")
            return
            
        plt.figure(figsize=figsize)
        
        # Get all documents
        doc_labels = list(self.data['wordcount'].keys())
        
        # Number of subplots needed
        n_docs = len(doc_labels)
        n_cols = min(3, n_docs)
        n_rows = (n_docs + n_cols - 1) // n_cols
        
        for i, label in enumerate(doc_labels):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Get top N words
            word_counts = self.data['wordcount'][label]
            top_words = word_counts.most_common(n)
            
            # Extract words and counts
            words, counts = zip(*top_words) if top_words else ([], [])
            
            # Create horizontal bar chart
            y_pos = np.arange(len(words))
            plt.barh(y_pos, counts, align='center')
            plt.yticks(y_pos, words)
            
            plt.title(f'Top {n} Words in {label}')
            plt.xlabel('Frequency')
            
        plt.tight_layout()
        plt.show()