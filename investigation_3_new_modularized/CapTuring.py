"""
File: CapTuring.py

Description: A reusable framework for comparative text analysis designed to analyze
relationships between documents with a focus on political spectrum positioning.
"""

from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


class StopWordsNotFoundError(Exception):
    """Custom exception raised when required stop words file is not found."""
    pass


class CapTuring:
    def __init__(self, custom_stop_words=None, require_stop_words=False):
        """Initialize the text analysis framework
        
        Args:
            custom_stop_words (list or str, optional): Custom stop words
            require_stop_words (bool): If True, raises error if stop words not found
        """
        self.data = defaultdict(dict)
        self.documents = {}
        self.baselines = {}
        self.document_groups = defaultdict(list)
        self.stop_words = set(ENGLISH_STOP_WORDS)
        
        # Add custom stop words if provided
        if custom_stop_words is not None:
            try:
                self.add_stop_words(custom_stop_words)
            except FileNotFoundError:
                if require_stop_words:
                    raise StopWordsNotFoundError(
                        f"Required stop words file not found: {custom_stop_words}. "
                        f"Please create a stop words file to improve analysis quality."
                    )
                else:
                    print(f"Warning: Stop words file not found. Using default stop words only.")
            
        self.update_vectorizer()
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def add_stop_words(self, stop_words):
        """Add custom stop words
        
        Args:
            stop_words (list or str): Either a list of stop words or a path to a file
        """
        if isinstance(stop_words, str) and os.path.isfile(stop_words):
            with open(stop_words, 'r', encoding='utf-8') as f:
                file_stop_words = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(file_stop_words)} stop words from {stop_words}")
            self.stop_words.update(file_stop_words)
        elif isinstance(stop_words, (list, set, tuple)):
            self.stop_words.update(stop_words)
        else:
            print(f"Invalid stop words format: {type(stop_words)}. Expected list or file path.")
            
        self.update_vectorizer()
        self._invalidate_matrices()
        
    def update_vectorizer(self):
        """Update the TF-IDF vectorizer with current stop words"""
        self.vectorizer = TfidfVectorizer(stop_words=list(self.stop_words))

    def simple_text_parser(self, filename):
        """Process simple, unformatted text documents"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
                
            words = re.findall(r'\b\w+\b', text.lower())
            word_count = Counter(words)
            
            return {
                'text': text,
                'wordcount': word_count,
                'numwords': len(words),
                'unique_words': len(word_count)
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return {'text': '', 'wordcount': Counter(), 'numwords': 0, 'unique_words': 0}

    def load_document(self, filename, label=None, is_baseline=False, baseline_type=None, group=None):
        """Register a document with the framework
        
        Args:
            filename (str): Path to the document file
            label (str, optional): Label for the document
            is_baseline (bool): Whether this document is a baseline
            baseline_type (str): Type of baseline (e.g., 'left', 'right', 'center')
            group (str, optional): Group to assign this document to (e.g., 'claude')
        """
        results = self.simple_text_parser(filename)
        
        if label is None:
            label = os.path.basename(filename)
            
        # Infer group from filename if not provided
        if group is None:
            group = os.path.basename(os.path.dirname(filename))

        # Store document and metadata
        self.documents[label] = results['text']
        self.document_groups[group].append(label)
        
        # Store all results in the data dictionary
        for k, v in results.items():
            self.data[k][label] = v
            
        # Register as baseline if specified
        if is_baseline and baseline_type is not None:
            self.baselines[baseline_type] = label
            print(f"Registered '{label}' as {baseline_type} baseline")

        self._invalidate_matrices()
        return label

    def _invalidate_matrices(self):
        """Reset computed matrices when documents change"""
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def _ensure_vectorized(self):
        """Ensure TF-IDF matrix is computed"""
        if self.tfidf_matrix is None and self.documents:
            labels = list(self.documents.keys())
            texts = [self.documents[label] for label in labels]
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            return labels
        return list(self.documents.keys())

    def calculate_similarity_matrix(self):
        """Calculate cosine similarity between all documents"""
        if not self.documents:
            print("No documents loaded")
            return None, []
            
        labels = self._ensure_vectorized()
        
        if self.similarity_matrix is None:
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        return self.similarity_matrix, labels

    def get_document_similarity(self, doc1_label, doc2_label):
        """Get cosine similarity between two specific documents"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        
        if sim_matrix is None:
            return None
            
        try:
            idx1 = labels.index(doc1_label)
            idx2 = labels.index(doc2_label)
            return sim_matrix[idx1, idx2]
        except ValueError:
            print(f"Document labels not found. Available labels: {labels}")
            return None

    def calculate_political_spectrum(self):
        """Calculate where each document falls on the political spectrum"""
        if len(self.baselines) < 2:
            print("Need at least two baseline documents")
            return {}
            
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            return {}
            
        spectrum_positions = {}
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
        
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        
        plt.title('Document Similarity Matrix')
        plt.tight_layout()
        plt.show()

    def visualize_political_spectrum(self, figsize=(12, 6), primary_axes=None):
        """Visualize documents on a political spectrum"""
        if len(self.baselines) < 2:
            print("Need at least two baseline documents")
            return
            
        baseline_types = list(self.baselines.keys())
        
        if primary_axes is None:
            primary_axes = baseline_types[:2]
        elif not all(axis in baseline_types for axis in primary_axes):
            print(f"Specified axes {primary_axes} not found in baselines {baseline_types}")
            primary_axes = baseline_types[:2]
            
        spectrum_positions = self.calculate_political_spectrum()
        
        # Create the main 2D plot
        self._create_2d_political_plot(spectrum_positions, primary_axes, figsize)
        
        # If we have more than 2 baselines, create radar plot
        if len(baseline_types) > 2:
            self._create_radar_plot(spectrum_positions, baseline_types)

    def _create_2d_political_plot(self, spectrum_positions, primary_axes, figsize=(12, 6)):
        """Create a 2D scatter plot for two selected baseline types"""
        x_values, y_values, annotations = [], [], []
        
        for label, positions in spectrum_positions.items():
            x = positions.get(primary_axes[0], 0)
            y = positions.get(primary_axes[1], 0)
            
            x_values.append(x)
            y_values.append(y)
            annotations.append(label)
        
        plt.figure(figsize=figsize)
        plt.scatter(x_values, y_values, c='blue', alpha=0.6)
        
        for i, label in enumerate(annotations):
            is_baseline = label in self.baselines.values()
            weight = 'bold' if is_baseline else 'normal'
            plt.annotate(label, (x_values[i], y_values[i]), 
                         fontweight=weight,
                         xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel(f'Similarity to {primary_axes[0]}')
        plt.ylabel(f'Similarity to {primary_axes[1]}')
        
        plt.title('Document Positioning on Political Spectrum')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def _create_radar_plot(self, spectrum_positions, baseline_types, figsize=(10, 8)):
        """Create a radar plot to visualize multi-dimensional baseline similarities"""
        import matplotlib.pyplot as plt
        
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
            
        # Set up the angles for each baseline
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)
        
        # Draw baseline labels
        plt.xticks(angles[:-1], baseline_types, color='black', size=8)
        
        # Draw y-axis labels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        # Plot each document
        for doc in non_baseline_docs:
            values = [spectrum_positions[doc].get(baseline, 0) for baseline in baseline_types]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=doc)
            ax.fill(angles, values, alpha=0.1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Multi-dimensional Political Spectrum Analysis')
        plt.show()

    def calculate_group_similarities(self):
        """Calculate average similarities between document groups"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            return None
            
        label_to_idx = {label: i for i, label in enumerate(labels)}
        group_similarities = defaultdict(dict)
        
        for group1 in self.document_groups:
            for group2 in self.document_groups:
                docs1 = [label_to_idx[doc] for doc in self.document_groups[group1] if doc in label_to_idx]
                docs2 = [label_to_idx[doc] for doc in self.document_groups[group2] if doc in label_to_idx]
                
                if docs1 and docs2:
                    # Calculate average similarity between groups
                    similarities = []
                    for i in docs1:
                        for j in docs2:
                            if i != j:  # Skip self-comparisons
                                similarities.append(sim_matrix[i, j])
                    
                    if similarities:
                        group_similarities[group1][group2] = np.mean(similarities)
        
        return group_similarities
    
    def get_group_to_baseline_similarities(self):
        """Calculate the similarity of each group to each baseline"""
        baseline_types = list(self.baselines.keys())
        baseline_labels = [self.baselines[bt] for bt in baseline_types]
        all_groups = list(self.document_groups.keys())
        
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            return {}
            
        label_to_idx = {label: i for i, label in enumerate(labels)}
        group_to_baseline = {}
        
        for group in all_groups:
            # Skip baseline groups
            if any(bl in self.document_groups[group] for bl in baseline_labels):
                continue
                
            group_to_baseline[group] = {}
            
            for baseline_type, baseline_label in self.baselines.items():
                if baseline_label in label_to_idx:
                    baseline_idx = label_to_idx[baseline_label]
                    
                    # Get all documents in this group
                    group_docs = [doc for doc in self.document_groups[group] if doc in label_to_idx]
                    group_indices = [label_to_idx[doc] for doc in group_docs]
                    
                    if group_indices:
                        # Calculate average similarity to the baseline
                        similarities = [sim_matrix[idx, baseline_idx] for idx in group_indices]
                        group_to_baseline[group][baseline_type] = np.mean(similarities)
        
        return group_to_baseline
    
    def normalize_baseline_similarities(self, similarities):
        """Convert similarities to percentages, ensuring they sum to 100%"""
        normalized = {}
        
        for group, values in similarities.items():
            total = sum(values.values())
            
            if total > 0:
                normalized[group] = {baseline: (sim / total) * 100 
                                  for baseline, sim in values.items()}
            else:
                normalized[group] = values
                
        return normalized

    #--------------------------------
    # Core Visualization Methods
    #--------------------------------
    
    def visualize_political_bias_distribution(self, figsize=(12, 7)):
        """Create a stacked horizontal bar chart showing political bias distribution of each LLM
        
        This visualization directly answers: "Which political position does each LLM lean toward?"
        """
        # Get similarities to baselines
        group_to_baseline = self.get_group_to_baseline_similarities()
        if not group_to_baseline:
            print("No groups to visualize")
            return
            
        # Normalize to get percentages
        normalized = self.normalize_baseline_similarities(group_to_baseline)
        
        # Prepare data
        baseline_types = list(self.baselines.keys())
        baseline_labels = set(self.baselines.values())
        
        # Filter to only LLM groups (not containing baseline documents)
        llm_groups = [group for group in normalized.keys() 
                    if not any(bl in self.document_groups[group] for bl in baseline_labels)]
        
        if not llm_groups:
            print("No LLM groups to visualize")
            return
        
        # Sort LLM groups alphabetically for consistent display
        llm_groups = sorted(llm_groups)
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create horizontal bar for each LLM
        y_pos = np.arange(len(llm_groups))
        
        # Choose distinct colors for political positions
        colors = plt.cm.tab10(np.linspace(0, 1, len(baseline_types)))
        
        # Initialize left positions for bar segments
        left = np.zeros(len(llm_groups))
        
        # Create a segment for each political position
        for i, baseline in enumerate(baseline_types):
            # Get percentage values for this baseline across all LLMs
            values = [normalized[group].get(baseline, 0) for group in llm_groups]
            
            # Create horizontal bars
            plt.barh(y_pos, values, left=left, color=colors[i], label=baseline)
            
            # Add percentage labels on the bars
            for j, value in enumerate(values):
                if value > 5:  # Only show label if segment is large enough
                    # Position text in the middle of the segment
                    text_x = left[j] + value/2
                    plt.text(text_x, y_pos[j], f"{value:.1f}%", 
                             ha='center', va='center', color='white', fontweight='bold')
            
            # Update left position for next segment
            left += values
        
        # Add labels and formatting
        plt.yticks(y_pos, llm_groups)
        plt.xlabel('Political Bias Distribution (%)')
        plt.title('Political Leaning of LLM Outputs', fontsize=14, pad=20)
        
        # Add legend at the top
        plt.legend(title='Political Position', bbox_to_anchor=(0.5, 1.05), 
                  loc='center', ncol=len(baseline_types))
        
        # Add grid lines for readability
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Ensure the x-axis goes from 0 to 100
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.show()
    
    def visualize_political_compass(self, figsize=(12, 10), min_similarity=0.3):
        """Create a political compass visualization showing LLMs in 2D political space
        
        This visualization directly answers: "How do different LLMs compare to each other
        in political positioning?"
        
        Args:
            figsize: Figure dimensions
            min_similarity: Minimum similarity to show connection between LLMs
        """
        if len(self.baselines) < 2:
            print("Need at least two baseline positions for political compass")
            return
            
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX library required. Install with: pip install networkx")
            return
            
        # Get the two most distinctive political positions as axes
        baseline_types = list(self.baselines.keys())
        
        # Select axes (try to use left/right if available)
        primary_axes = []
        if "left" in baseline_types and "right" in baseline_types:
            primary_axes = ["left", "right"]
        else:
            primary_axes = baseline_types[:2]
            
        # Get similarities to baselines
        group_to_baseline = self.get_group_to_baseline_similarities()
        if not group_to_baseline:
            print("No groups to visualize")
            return
            
        # Filter to just LLM groups (no baseline documents)
        baseline_labels = set(self.baselines.values())
        llm_groups = {}
        
        for group, similarities in group_to_baseline.items():
            if not any(bl in self.document_groups[group] for bl in baseline_labels):
                if primary_axes[0] in similarities and primary_axes[1] in similarities:
                    llm_groups[group] = similarities
        
        if not llm_groups:
            print("No LLM groups to visualize")
            return
            
        # Create the figure
        plt.figure(figsize=figsize)
        
        # Create a graph for LLM relationships
        G = nx.Graph()
        
        # Add nodes (LLMs)
        pos = {}  # Positions based on political coordinates
        for group, similarities in llm_groups.items():
            # Use similarity to each baseline as coordinates
            x = similarities[primary_axes[0]]
            y = similarities[primary_axes[1]]
            
            # Add to graph
            G.add_node(group, size=len(self.document_groups[group]))
            
            # Store position
            pos[group] = (x, y)
            
        # Add edges (similarities between LLMs)
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is not None:
            label_to_idx = {label: i for i, label in enumerate(labels)}
            
            for group1 in llm_groups:
                for group2 in llm_groups:
                    if group1 != group2:
                        # Calculate average similarity between groups
                        similarities = []
                        
                        # Get document indices
                        docs1 = [label_to_idx[doc] for doc in self.document_groups[group1] 
                               if doc in label_to_idx]
                        docs2 = [label_to_idx[doc] for doc in self.document_groups[group2] 
                               if doc in label_to_idx]
                        
                        # Calculate similarities
                        for i in docs1:
                            for j in docs2:
                                similarities.append(sim_matrix[i, j])
                        
                        if similarities:
                            avg_sim = np.mean(similarities)
                            
                            # Add edge if similarity is above threshold
                            if avg_sim >= min_similarity:
                                G.add_edge(group1, group2, weight=avg_sim)
        
        # Draw nodes sized by number of documents
        node_sizes = [G.nodes[n]['size'] * 150 for n in G.nodes()]
        
        # Draw the LLM nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8)
        
        # Draw edges with width based on similarity
        for u, v, data in G.edges(data=True):
            width = data['weight'] * 5
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                 width=width, alpha=0.5, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
        
        # Add political baseline markers
        plt.scatter([0, 1], [0, 1], color='red', marker='*', s=300, alpha=0.7)
        
        # Add baseline labels
        plt.text(0, 0, primary_axes[0], fontsize=14, ha='right', va='top', 
                color='darkred', fontweight='bold')
        plt.text(1, 1, primary_axes[1], fontsize=14, ha='left', va='bottom', 
                color='darkred', fontweight='bold')
        
        # Add arrows showing political dimensions
        plt.arrow(-0.05, -0.05, 1.1, 0, head_width=0.03, head_length=0.03, 
                 fc='darkred', ec='darkred', alpha=0.4)
        plt.arrow(-0.05, -0.05, 0, 1.1, head_width=0.03, head_length=0.03, 
                 fc='darkred', ec='darkred', alpha=0.4)
        
        # Add title
        plt.title(f'Political Compass: {primary_axes[0]} vs {primary_axes[1]}', 
                 fontsize=16, pad=20)
        
        # Add explanatory text
        plt.annotate('Node size = number of documents\nEdge width = similarity between LLMs',
                   xy=(0.05, 0.05), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Adjust axes
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        
        # Format with gridlines
        plt.grid(linestyle='--', alpha=0.3)
        
        # Add labels
        plt.xlabel(f'Similarity to {primary_axes[0]}', fontsize=12)
        plt.ylabel(f'Similarity to {primary_axes[1]}', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
    def visualize_key_word_comparison(self, n_words=15, figsize=(14, 10)):
        """Create a comparison of politically distinctive words across document groups
        
        This visualization answers: "Which words make these documents politically distinctive?"
        """
        if not self.data['wordcount']:
            print("No word count data available")
            return
        
        # Get baseline groups
        baseline_types = list(self.baselines.keys())
        baseline_labels = [self.baselines[bt] for bt in baseline_types]
        
        # Group word counts by baseline type
        baseline_words = {}
        for i, baseline_type in enumerate(baseline_types):
            baseline_label = baseline_labels[i]
            
            # Get the group for this baseline
            baseline_group = None
            for group, docs in self.document_groups.items():
                if baseline_label in docs:
                    baseline_group = group
                    break
            
            if baseline_group:
                # Combine word counts for all documents in this group
                combined_counts = Counter()
                for doc in self.document_groups[baseline_group]:
                    if doc in self.data['wordcount']:
                        combined_counts.update(self.data['wordcount'][doc])
                
                # Filter out stop words
                filtered_counts = Counter({word: count for word, count in combined_counts.items()
                                        if word.lower() not in self.stop_words})
                
                baseline_words[baseline_type] = filtered_counts
        
        # Get LLM groups (excluding baselines)
        llm_groups = []
        for group in self.document_groups:
            if not any(bl in self.document_groups[group] for bl in baseline_labels):
                llm_groups.append(group)
        
        # Combine similar LLM groups (if there are too many)
        max_llms_to_show = 3  # Limit the number of LLMs to display
        if len(llm_groups) > max_llms_to_show:
            # Take the first few LLMs
            llm_groups = sorted(llm_groups)[:max_llms_to_show]
            print(f"Showing only the first {max_llms_to_show} LLM groups: {', '.join(llm_groups)}")
        
        # Get word counts for LLM groups
        llm_words = {}
        for llm_group in llm_groups:
            # Combine word counts for all documents in this group
            combined_counts = Counter()
            for doc in self.document_groups[llm_group]:
                if doc in self.data['wordcount']:
                    combined_counts.update(self.data['wordcount'][doc])
            
            # Filter out stop words
            filtered_counts = Counter({word: count for word, count in combined_counts.items()
                                     if word.lower() not in self.stop_words})
            
            llm_words[llm_group] = filtered_counts
        
        # Calculate TF-IDF to find distinctive words for each group
        all_words = set()
        for counts in baseline_words.values():
            all_words.update(counts.keys())
        for counts in llm_words.values():
            all_words.update(counts.keys())
        
        # Build a document-term matrix
        all_groups = list(baseline_words.keys()) + list(llm_words.keys())
        word_list = sorted(list(all_words))
        
        # Create term frequency matrix
        tf_matrix = np.zeros((len(all_groups), len(word_list)))
        
        # Fill matrix with term frequencies
        for i, group in enumerate(all_groups):
            if i < len(baseline_words):
                # This is a baseline group
                counts = baseline_words[group]
            else:
                # This is an LLM group
                counts = llm_words[all_groups[i]]
                
            for j, word in enumerate(word_list):
                tf_matrix[i, j] = counts.get(word, 0)
        
        # Normalize term frequencies
        row_sums = tf_matrix.sum(axis=1, keepdims=True)
        tf_matrix = np.divide(tf_matrix, row_sums, out=np.zeros_like(tf_matrix), where=row_sums!=0)
        
        # Find distinctive words for each group
        distinctive_words = {}
        
        for i, group in enumerate(all_groups):
            # Calculate distinctiveness score
            # Higher score = more unique to this group
            word_scores = {}
            
            for j, word in enumerate(word_list):
                # Skip words that don't appear in this group
                if tf_matrix[i, j] == 0:
                    continue
                    
                # Calculate how distinctive this word is for this group
                # High value in this group / average value in other groups
                this_group_tf = tf_matrix[i, j]
                other_groups_tf = [tf_matrix[k, j] for k in range(len(all_groups)) if k != i]
                
                if other_groups_tf:
                    avg_other_tf = sum(other_groups_tf) / len(other_groups_tf)
                    if avg_other_tf > 0:
                        distinctiveness = this_group_tf / avg_other_tf
                    else:
                        distinctiveness = this_group_tf * 100  # Very distinctive (not in other groups)
                else:
                    distinctiveness = this_group_tf * 100
                
                # Also consider absolute frequency (weight = frequency * distinctiveness)
                original_count = 0
                if i < len(baseline_words):
                    original_count = baseline_words[group].get(word, 0)
                else:
                    original_count = llm_words[all_groups[i]].get(word, 0)
                
                # Calculate final score (balance frequency and distinctiveness)
                score = np.log1p(original_count) * distinctiveness
                word_scores[word] = score
            
            # Get top distinctive words
            distinctive_words[group] = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n_words]
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Number of groups to show
        n_groups = len(all_groups)
        
        # Calculate grid layout
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        # Create a subplot for each group
        for i, group in enumerate(all_groups):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Get distinctive words and scores
            words_and_scores = distinctive_words[group]
            words = [w for w, _ in words_and_scores]
            scores = [s for _, s in words_and_scores]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(words))
            
            # Use appropriate color for each group
            if i < len(baseline_words):
                # Baseline group - use political colors
                color = plt.cm.tab10(i / len(baseline_types))
            else:
                # LLM group - use blue
                color = 'skyblue'
                
            plt.barh(y_pos, scores, align='center', color=color, alpha=0.7)
            plt.yticks(y_pos, words)
            
            # Format
            if i < len(baseline_words):
                plt.title(f'Distinctive Words: {group} baseline', fontsize=12)
            else:
                plt.title(f'Distinctive Words: {group}', fontsize=12)
                
            # Remove x-labels (scores aren't meaningful on their own)
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        plt.suptitle('Politically Distinctive Words by Group', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        plt.show()
        
    def visualize_consistency_across_documents(self, figsize=(12, 8)):
        """Visualize consistency of political leaning across multiple documents from the same LLM
        
        This visualization answers: "How consistent is each LLM in its political leaning?"
        """
        # Get baseline types and labels
        baseline_types = list(self.baselines.keys())
        baseline_labels = [self.baselines[bt] for bt in baseline_types]
        
        # Get similarity matrix
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            print("No similarity matrix available")
            return
            
        # Get mapping from label to index
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        # Get LLM groups (excluding baselines)
        llm_groups = []
        for group in self.document_groups:
            if not any(bl in self.document_groups[group] for bl in baseline_labels):
                # Only include groups with multiple documents
                if len(self.document_groups[group]) > 1:
                    llm_groups.append(group)
        
        if not llm_groups:
            print("No LLM groups with multiple documents to visualize")
            return
            
        # Calculate similarity statistics for each LLM group to each baseline
        consistency_data = {}
        
        for llm_group in llm_groups:
            consistency_data[llm_group] = {}
            
            for baseline_type, baseline_label in self.baselines.items():
                if baseline_label in label_to_idx:
                    baseline_idx = label_to_idx[baseline_label]
                    
                    # Get similarities for all documents in this group
                    similarities = []
                    
                    for doc in self.document_groups[llm_group]:
                        if doc in label_to_idx:
                            doc_idx = label_to_idx[doc]
                            similarities.append(sim_matrix[doc_idx, baseline_idx])
                    
                    if similarities:
                        # Calculate statistics
                        consistency_data[llm_group][baseline_type] = {
                            'mean': np.mean(similarities),
                            'std': np.std(similarities),
                            'min': np.min(similarities),
                            'max': np.max(similarities),
                            'range': np.max(similarities) - np.min(similarities),
                            'individual': similarities
                        }
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Set up bar positions
        n_llms = len(llm_groups)
        n_baselines = len(baseline_types)
        bar_width = 0.7 / n_baselines
        
        # Set up x positions for bars
        index = np.arange(n_llms)
        
        # Create a bar for each baseline for each LLM
        for i, baseline_type in enumerate(baseline_types):
            means = []
            errors = []
            
            for llm_group in llm_groups:
                if baseline_type in consistency_data[llm_group]:
                    means.append(consistency_data[llm_group][baseline_type]['mean'])
                    errors.append(consistency_data[llm_group][baseline_type]['std'])
                else:
                    means.append(0)
                    errors.append(0)
            
            # Calculate bar positions
            bar_pos = index + i * bar_width - 0.7/2 + bar_width/2
            
            # Plot bars with error bars
            plt.bar(bar_pos, means, bar_width, alpha=0.7, 
                   label=baseline_type, color=plt.cm.tab10(i / n_baselines))
            
            # Add error bars
            plt.errorbar(bar_pos, means, yerr=errors, fmt='none', ecolor='black', capsize=3)
        
        # Format plot
        plt.xlabel('LLM Group')
        plt.ylabel('Mean Similarity Score (with Standard Deviation)')
        plt.title('Consistency of Political Leaning Across Documents', fontsize=14)
        plt.xticks(index, llm_groups)
        plt.legend(title='Political Position')
        
        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def visualize_all_analyses(self):
        """Run all core visualizations sequentially for comprehensive analysis"""
        print("1. Political Bias Distribution...")
        self.visualize_political_bias_distribution()
        
        print("\n2. Political Compass...")
        self.visualize_political_compass()
        
        print("\n3. Politically Distinctive Words...")
        self.visualize_key_word_comparison()
        
        print("\n4. Political Consistency Analysis...")
        self.visualize_consistency_across_documents()
        
        print("\nAnalysis visualizations complete!")
        
    def compare_word_frequencies(self, n=20, figsize=(15, 8), exclude_stop_words=True):
        """Compare most common words across documents"""
        if not self.data['wordcount']:
            print("No word count data available")
            return
            
        plt.figure(figsize=figsize)
        
        doc_labels = list(self.data['wordcount'].keys())
        n_docs = len(doc_labels)
        n_cols = min(3, n_docs)
        n_rows = (n_docs + n_cols - 1) // n_cols
        
        for i, label in enumerate(doc_labels):
            plt.subplot(n_rows, n_cols, i+1)
            
            word_counts = self.data['wordcount'][label]
            
            # Filter out stop words if requested
            if exclude_stop_words and self.stop_words:
                filtered_counts = Counter({word: count for word, count 
                                         in word_counts.items() 
                                         if word.lower() not in self.stop_words})
                top_words = filtered_counts.most_common(n)
            else:
                top_words = word_counts.most_common(n)
            
            # Extract words and counts
            words, counts = zip(*top_words) if top_words else ([], [])
            
            # Create horizontal bar chart
            y_pos = np.arange(len(words))
            plt.barh(y_pos, counts, align='center')
            plt.yticks(y_pos, words)
            
            title = f'Top {n} Words in {label}'
            if exclude_stop_words:
                title += ' (Stop Words Excluded)'
            plt.title(title)
            plt.xlabel('Frequency')
            
        plt.tight_layout()
        plt.show()
        
    def get_stop_words(self):
        """Return the current set of stop words"""
        return self.stop_words
        
    def export_stop_words(self, filename):
        """Export the current set of stop words to a file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for word in sorted(self.stop_words):
                    f.write(f"{word}\n")
            print(f"Exported {len(self.stop_words)} stop words to {filename}")
        except Exception as e:
            print(f"Error exporting stop words to {filename}: {e}")