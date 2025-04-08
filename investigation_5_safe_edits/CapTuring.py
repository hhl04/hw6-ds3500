"""
File: CapTuring.py

Description: A streamlined framework for comparative text analysis
designed to analyze and visualize relationships between documents,
with enhanced support for document grouping and political bias analysis.
"""

from collections import Counter, defaultdict
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import scipy.cluster.hierarchy as sch
import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CapTuring:
    """
    CapTuring: A framework for comparative text analysis.
    
    This class provides methods to load, analyze, and visualize text documents,
    with a focus on comparison between different perspectives and analysis of
    political bias in language models.
    """

    def __init__(self, vectorizer=None, **kwargs):
        """Constructor"""
        self.data = defaultdict(dict)
        self.documents = {}  # Store raw text of documents
        self.baselines = {}  # Store baseline documents
        
        # Configure vectorizer
        vectorizer_kwargs = kwargs.get('vectorizer_kwargs', {})
        self.vectorizer = vectorizer or TfidfVectorizer(stop_words='english', **vectorizer_kwargs)
        self.parsers = {'simple': self.simple_text_parser}
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
        # Configuration
        self.config = {
            'similarity_metric': kwargs.get('similarity_metric', 'cosine'),
            'min_word_length': kwargs.get('min_word_length', 2),
            'case_sensitive': kwargs.get('case_sensitive', False),
            'default_colors': {
                'left': '#1f77b4',      # Blue for left
                'right': '#d62728',     # Red for right
                'center': '#2ca02c',    # Green for center
            }
        }

    def register_parser(self, parser_name, parser_function):
        """
        Register a custom parser function
        
        Args:
            parser_name: Name to identify the parser
            parser_function: Function that takes a filename and returns a dict with text data
            
        Returns:
            self: For method chaining
        """
        if not callable(parser_function):
            raise ValueError("Parser function must be callable")
            
        self.parsers[parser_name] = parser_function
        return self

    def simple_text_parser(self, filename):
        """Parse a simple text file"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
            words = self.tokenize_text(text)
            word_count = Counter(words)
            
            results = {
                'text': text,
                'wordcount': word_count,
                'numwords': len(words),
                'unique_words': len(word_count)
            }
            
            results.update(self.calculate_advanced_features(text, words))
            return results
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return {'text': '', 'wordcount': Counter(), 'numwords': 0, 'unique_words': 0}

    def tokenize_text(self, text):
        """Tokenize text into words"""
        if not self.config['case_sensitive']:
            text = text.lower()
            
        min_length = self.config['min_word_length']
        words = re.findall(rf'\b\w{{{min_length},}}\b', text)
        
        # Filter out stop words
        if hasattr(self.vectorizer, 'stop_words_'):
            words = [word for word in words if word not in self.vectorizer.stop_words_]
            
        return words
    
    def clean_text(self, text, **kwargs):
        """Clean and normalize text data"""
        # Remove URLs
        if kwargs.get('remove_urls', True):
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove HTML tags
        if kwargs.get('remove_html', True):
            text = re.sub(r'<.*?>', ' ', text)
            
        # Remove punctuation
        if kwargs.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # Remove numbers
        if kwargs.get('remove_numbers', False):
            text = re.sub(r'\d+', ' ', text)
            
        # Normalize whitespace
        if kwargs.get('remove_extra_whitespace', True):
            text = re.sub(r'\s+', ' ', text).strip()
            
        return text
    
    def load_text(self, source, label=None, **kwargs):
        """Register a document with the framework"""
        # Extract common kwargs
        is_baseline = kwargs.get('is_baseline', False)
        baseline_type = kwargs.get('baseline_type')
        metadata = kwargs.get('metadata', {})
        clean_text_options = kwargs.get('clean_text_options', {})
        
        # Determine source type and generate label
        is_file = isinstance(source, str) and (
            source.endswith('.txt') or source.endswith('.md') or 
            source.endswith('.html') or '/' in source or '\\' in source
        )
        
        if label is None:
            label = kwargs.get('label_prefix', '') + source if is_file else f"doc_{len(self.documents) + 1}"
        
        if is_file:
            parser_name = kwargs.get('parser_name', 'simple')
            parser_func = self.parsers.get(parser_name, self.simple_text_parser)
            results = parser_func(source)
            
            # Clean the text
            if 'text' in results:
                results['text'] = self.clean_text(results['text'], **clean_text_options)
                words = self.tokenize_text(results['text'])
                results['wordcount'] = Counter(words)
                results['numwords'] = len(words)
                results['unique_words'] = len(results['wordcount'])
                results.update(self.calculate_advanced_features(results['text'], words))
        else:
            text = self.clean_text(source, **clean_text_options)
            words = self.tokenize_text(text)
            
            results = {
                'text': text,
                'wordcount': Counter(words),
                'numwords': len(words),
                'unique_words': len(set(words))
            }
            results.update(self.calculate_advanced_features(text, words))

        # Store document and data
        self.documents[label] = results['text']
        
        for k, v in results.items():
            self.data[k][label] = v
            
        if metadata:
            if 'metadata' not in self.data:
                self.data['metadata'] = {}
            self.data['metadata'][label] = metadata
            
        # Register as baseline
        if is_baseline and baseline_type:
            self.baselines[baseline_type] = label
            print(f"Registered '{label}' as {baseline_type} baseline")

        # Reset matrices
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
        return label

    def add_stop_words(self, stop_words_list):
        """Add custom stop words to the vectorizer"""
        if not stop_words_list:
            return self
            
        # Initialize stop_words_ attribute if needed
        if not hasattr(self.vectorizer, 'stop_words_'):
            self.vectorizer.stop_words_ = set() if not hasattr(self.vectorizer, 'get_stop_words') else set(self.vectorizer.get_stop_words())
                
        # Add new stop words
        self.vectorizer.stop_words_.update(stop_words_list)
        return self
        
    def load_stop_words(self, stopfile, **kwargs):
        """Load custom stop words from a file"""
        verbose = kwargs.get('verbose', True)
        stop_words = []
        
        try:
            with open(stopfile, 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        stop_words.append(word)
            
            self.add_stop_words(stop_words)
            
            if verbose:
                print(f"Loaded {len(stop_words)} stop words from {stopfile}")
            return stop_words
        except Exception as e:
            if verbose:
                print(f"Error loading stop words from {stopfile}: {e}")
            return []

    def calculate_similarity_matrix(self):
        """Calculate cosine similarity between all documents"""
        if not self.documents:
            return None, []
            
        # Create TF-IDF matrix if needed
        if self.tfidf_matrix is None:
            labels = list(self.documents.keys())
            texts = [self.documents[label] for label in labels]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            labels = list(self.documents.keys())
        
        # Calculate similarity matrix if needed
        if self.similarity_matrix is None:
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
        return self.similarity_matrix, labels

    def get_document_similarity(self, doc1_label, doc2_label):
        """Get similarity between two documents"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        
        if sim_matrix is None:
            return None
            
        try:
            idx1 = labels.index(doc1_label)
            idx2 = labels.index(doc2_label)
            return sim_matrix[idx1, idx2]
        except ValueError:
            print(f"Document labels not found")
            return None

    def get_most_similar_documents(self, doc_label, n=5):
        """Find most similar documents to a given document"""
        sim_matrix, labels = self.calculate_similarity_matrix()
        
        if sim_matrix is None or doc_label not in labels:
            return []
            
        idx = labels.index(doc_label)
        similarities = sim_matrix[idx]
        
        # Create list of (label, similarity) tuples and sort
        similar_docs = [(labels[i], similarities[i]) for i in range(len(labels))]
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove self from results
        similar_docs = [doc for doc in similar_docs if doc[0] != doc_label]
                
        return similar_docs[:n]
        
    def get_documents_by_group(self, group_name=None, include_group_docs=True, include_individual_docs=True):
        """Get all documents that belong to a specific group
        
        Args:
            group_name: Group identifier or None for all groups
            include_group_docs: Include combined group documents
            include_individual_docs: Include individual documents
            
        Returns:
            dict: Dictionary of documents filtered by group
        """
        if 'metadata' not in self.data:
            return {}
            
        result = {}
        
        for label, text in self.documents.items():
            # Skip if document doesn't have metadata
            if label not in self.data['metadata']:
                continue
                
            metadata = self.data['metadata'][label]
            
            # Check if document has group information
            if 'group' not in metadata:
                continue
                
            doc_group = metadata['group']
            
            # Filter by group name if specified
            if group_name is not None and doc_group != group_name:
                continue
                
            # Filter by document type
            is_group_doc = metadata.get('is_group', False)
            
            if (is_group_doc and include_group_docs) or (not is_group_doc and include_individual_docs):
                result[label] = text
                
        return result

    def get_group_document(self, group_name):
        """Get the combined group document if it exists
        
        Args:
            group_name: Group identifier
            
        Returns:
            str: Document label for the combined group document or None
        """
        if 'metadata' not in self.data:
            return None
            
        for label in self.documents:
            if label in self.data['metadata']:
                metadata = self.data['metadata'][label]
                
                # Check if this is a group document with the right group
                if (metadata.get('group') == group_name and 
                    metadata.get('is_group', False) == True):
                    return label
                    
        return None
        
    def get_group_labels(self):
        """Get all unique group labels from metadata
        
        Returns:
            list: List of unique group names
        """
        if 'metadata' not in self.data:
            return []
            
        groups = set()
        
        for label, metadata in self.data['metadata'].items():
            if 'group' in metadata:
                groups.add(metadata['group'])
                
        return sorted(list(groups))
        
    def get_baseline_documents(self, **kwargs):
        """Get all baseline documents
        
        Args:
            **kwargs: Additional options including:
                - types_only: Return only baseline types (default: False)
                - include_text: Include document text (default: False)
            
        Returns:
            dict: Dictionary mapping baseline types to document labels or data
        """
        if kwargs.get('types_only', False):
            return list(self.baselines.keys())
            
        if kwargs.get('include_text', False):
            result = {}
            for btype, label in self.baselines.items():
                result[btype] = {
                    'label': label,
                    'text': self.get_document_text(label)
                }
            return result
            
        return self.baselines.copy()
        
    def get_document_text(self, label):
        """Get text for a specific document
        
        Args:
            label: Document label
            
        Returns:
            str: Document text or empty string if not found
        """
        return self.documents.get(label, "")
        
    def calculate_document_positions(self, baseline_types=None):
        """Calculate document positions relative to baselines"""
        # Use all baselines if none specified
        if baseline_types is None:
            baseline_types = list(self.baselines.keys())
            
        if not baseline_types:
            print("No baseline documents specified")
            return {}
            
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            return {}
            
        positions = {}
        
        # Get indices of baseline documents
        baseline_indices = {
            btype: labels.index(self.baselines[btype]) 
            for btype in baseline_types if btype in self.baselines
            and self.baselines[btype] in labels
        }
        
        # Calculate positions
        for i, label in enumerate(labels):
            doc_positions = {
                btype: sim_matrix[i, idx] for btype, idx in baseline_indices.items()
            }
            positions[label] = doc_positions
            
        return positions

    def calculate_advanced_features(self, text, words):
        """Calculate advanced text features"""
        features = {}
        
        # Calculate average word length
        features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
            
        # Calculate lexical diversity
        features['lexical_diversity'] = len(set(words)) / len(words) if words else 0
            
        # Calculate average sentence length
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if sentences:
            sentence_lengths = [len(re.findall(r'\b\w+\b', s.lower())) for s in sentences]
            features['avg_sentence_length'] = sum(sentence_lengths) / len(sentences)
            features['num_sentences'] = len(sentences)
        else:
            features['avg_sentence_length'] = 0
            features['num_sentences'] = 0
            
        return features

    def visualize_similarity_heatmap(self, **kwargs):
        """
        Create a heatmap visualization of document similarities with baselines
        
        Args:
            **kwargs: Additional options including:
                - figsize: Figure size as (width, height) tuple
                - title: Plot title
                - show: Whether to display the plot (default: True)
                - use_clustering: Whether to use hierarchical clustering (default: True)
                - annotate: Whether to show numeric similarity values (default: True)
                - cmap: Colormap to use (default: 'YlOrRd')
                - llm_filter: List of LLM names to include (default: all)
                - baseline_filter: List of baseline types to include (default: all)
                - use_combined: Whether to use combined documents for LLMs with multiple docs (default: True)
                
        Returns:
            matplotlib Figure object
        """
        # Calculate similarities for all documents
        sim_matrix, labels = self.calculate_similarity_matrix()
        if sim_matrix is None:
            print("No documents to visualize")
            return None
            
        # Get baseline document labels and types
        baseline_types = kwargs.get('baseline_filter', list(self.baselines.keys()))
        baseline_labels = [self.baselines[btype] for btype in baseline_types if btype in self.baselines]
        
        # Get LLM document labels based on filter and combined preference
        use_combined = kwargs.get('use_combined', True)
        llm_filter = kwargs.get('llm_filter')
        
        llm_docs = {}
        if 'metadata' in self.data:
            for label in labels:
                if label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    
                    # Filter by type and group
                    if metadata.get('type') == 'llm':
                        group = metadata.get('group')
                        
                        # Filter by LLM name if specified
                        if llm_filter is not None and group not in llm_filter:
                            continue
                            
                        # Prefer combined documents if available
                        if use_combined and metadata.get('is_group', False):
                            llm_docs[group] = label
                        elif not use_combined and not metadata.get('is_group', False):
                            # For non-combined, use first document of each LLM
                            if group not in llm_docs:
                                llm_docs[group] = label
        
        # Convert to list of labels
        llm_labels = list(llm_docs.values())
        
        # Create subset of similarity matrix for visualization
        row_indices = [labels.index(label) for label in llm_labels]
        col_indices = [labels.index(label) for label in baseline_labels]
        
        heatmap_data = sim_matrix[np.ix_(row_indices, col_indices)]
        
        # Set up figure
        figsize = kwargs.get('figsize', (10, 8))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use hierarchical clustering if requested
        use_clustering = kwargs.get('use_clustering', True)
        
        if use_clustering and len(llm_labels) > 1:
            # Cluster rows (LLMs)
            row_linkage = sch.linkage(heatmap_data, method='average')
            row_dendrogram = sch.dendrogram(row_linkage, ax=ax, no_plot=True)
            row_idx = row_dendrogram['leaves']
            
            # Reorder data
            heatmap_data = heatmap_data[row_idx, :]
            llm_labels = [llm_labels[i] for i in row_idx]
        
        # Create heatmap
        cmap = kwargs.get('cmap', 'YlOrRd')
        im = ax.imshow(heatmap_data, cmap=cmap)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score')
        
        # Set tick labels
        ax.set_xticks(np.arange(len(baseline_labels)))
        ax.set_yticks(np.arange(len(llm_labels)))
        
        # Clean up baseline labels for display
        display_baseline_labels = [label.replace('human_', '').capitalize() for label in baseline_types]
        
        # Clean up LLM labels for display
        display_llm_labels = []
        for label in llm_labels:
            if 'metadata' in self.data and label in self.data['metadata']:
                model = self.data['metadata'][label].get('model', label)
                display_llm_labels.append(model.capitalize())
            else:
                display_llm_labels.append(label)
        
        ax.set_xticklabels(display_baseline_labels, rotation=45, ha='right')
        ax.set_yticklabels(display_llm_labels)
        
        # Add annotations
        if kwargs.get('annotate', True):
            for i in range(len(llm_labels)):
                for j in range(len(baseline_labels)):
                    text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                                  ha="center", va="center", color="black" if heatmap_data[i, j] < 0.7 else "white")
        
        # Add title
        title = kwargs.get('title', 'Political Bias Similarity Heatmap')
        ax.set_title(title)
        
        # Improve layout
        fig.tight_layout()
        
        # Show figure
        if kwargs.get('show', True):
            plt.show()
            
        return fig

    def visualize_most_similar_baseline(self, **kwargs):
        """
        Create a visualization showing each LLM's most similar baseline
        
        Args:
            **kwargs: Additional options including:
                - figsize: Figure size as (width, height) tuple
                - title: Plot title
                - show: Whether to display the plot (default: True)
                - llm_filter: List of LLM names to include (default: all)
                - use_combined: Whether to use combined documents for LLMs with multiple docs (default: True)
                
        Returns:
            matplotlib Figure object
        """
        # Calculate positions for all documents
        positions = self.calculate_document_positions()
        if not positions:
            print("No document positions to visualize")
            return None
            
        # Get LLM document labels based on filter and combined preference
        use_combined = kwargs.get('use_combined', True)
        llm_filter = kwargs.get('llm_filter')
        
        llm_docs = {}
        if 'metadata' in self.data:
            for label in positions.keys():
                if label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    
                    # Filter by type and group
                    if metadata.get('type') == 'llm':
                        group = metadata.get('group')
                        
                        # Filter by LLM name if specified
                        if llm_filter is not None and group not in llm_filter:
                            continue
                            
                        # Prefer combined documents if available
                        if use_combined and metadata.get('is_group', False):
                            llm_docs[group] = label
                        elif not use_combined and not metadata.get('is_group', False):
                            # For non-combined, use first document of each LLM
                            if group not in llm_docs:
                                llm_docs[group] = label
        
        # Set up figure
        figsize = kwargs.get('figsize', (12, len(llm_docs) * 1.5))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for visualization
        llm_names = []
        most_similar_baseline = []
        similarity_scores = []
        baseline_colors = []
        
        for llm_name, label in llm_docs.items():
            if label in positions:
                # Find most similar baseline
                pos = positions[label]
                max_baseline = max(pos.items(), key=lambda x: x[1])
                
                llm_names.append(llm_name.upper())
                most_similar_baseline.append(max_baseline[0].capitalize())
                similarity_scores.append(max_baseline[1])
                
                # Get color for baseline
                color = self.config['default_colors'].get(max_baseline[0], '#777777')
                baseline_colors.append(color)
        
        # Create horizontal bars
        y_pos = np.arange(len(llm_names))
        bars = ax.barh(y_pos, similarity_scores, color=baseline_colors)
        
        # Add baseline names and similarity scores to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width / 2, bar.get_y() + bar.get_height() / 2,
                    f"{most_similar_baseline[i]}: {similarity_scores[i]:.2f}",
                    ha='center', va='center', color='white', fontweight='bold')
        
        # Set axis labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(llm_names)
        ax.set_xlabel('Similarity Score')
        
        title = kwargs.get('title', 'Most Similar Political Baseline for Each LLM')
        ax.set_title(title)
        
        # Set x-axis limits
        ax.set_xlim(0, 1)
        
        # Add legend
        legend_handles = []
        legend_labels = []
        
        for baseline_type, color in self.config['default_colors'].items():
            if baseline_type in self.baselines:
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append(baseline_type.capitalize())
        
        ax.legend(legend_handles, legend_labels, loc='lower right')
        
        # Improve layout
        fig.tight_layout()
        
        # Show figure
        if kwargs.get('show', True):
            plt.show()
            
        return fig

    def visualize_political_bias_bars(self, **kwargs):
        """
        Create horizontal stacked bars showing percentage breakdown of political leanings
        
        Args:
            **kwargs: Additional options including:
                - figsize: Figure size as (width, height) tuple
                - title: Plot title
                - show: Whether to display the plot (default: True)
                - llm_filter: List of LLM names to include (default: all)
                - use_combined: Whether to use combined documents for LLMs with multiple docs (default: True)
                - sort_by: Baseline type to sort by (default: first baseline)
                
        Returns:
            matplotlib Figure object
        """
        # Calculate positions for all documents
        positions = self.calculate_document_positions()
        if not positions:
            print("No document positions to visualize")
            return None
            
        # Get baseline types
        baseline_types = list(self.baselines.keys())
        if not baseline_types:
            print("No baseline types defined")
            return None
            
        # Get LLM document labels based on filter and combined preference
        use_combined = kwargs.get('use_combined', True)
        llm_filter = kwargs.get('llm_filter')
        
        llm_docs = {}
        if 'metadata' in self.data:
            for label in positions.keys():
                if label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    
                    # Filter by type and group
                    if metadata.get('type') == 'llm':
                        group = metadata.get('group')
                        
                        # Filter by LLM name if specified
                        if llm_filter is not None and group not in llm_filter:
                            continue
                            
                        # Prefer combined documents if available
                        if use_combined and metadata.get('is_group', False):
                            llm_docs[group] = label
                        elif not use_combined and not metadata.get('is_group', False):
                            # For non-combined, use first document of each LLM
                            if group not in llm_docs:
                                llm_docs[group] = label
        
        # Prepare data for visualization
        llm_names = []
        bias_data = {baseline: [] for baseline in baseline_types}
        
        for llm_name, label in llm_docs.items():
            if label in positions:
                llm_names.append(llm_name.upper())
                
                # Get baseline similarities and normalize to sum to 100%
                pos = positions[label]
                total = sum(pos.values())
                
                for baseline in baseline_types:
                    if baseline in pos:
                        normalized_value = (pos[baseline] / total) * 100
                        bias_data[baseline].append(normalized_value)
                    else:
                        bias_data[baseline].append(0)
        
        # Sort data by specified baseline if requested
        sort_by = kwargs.get('sort_by')
        if sort_by in baseline_types:
            sort_indices = np.argsort(bias_data[sort_by])[::-1]  # Descending order
            
            # Apply sorting to all data
            llm_names = [llm_names[i] for i in sort_indices]
            for baseline in baseline_types:
                bias_data[baseline] = [bias_data[baseline][i] for i in sort_indices]
        
        # Set up figure
        figsize = kwargs.get('figsize', (12, len(llm_names) * 0.8))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create stacked horizontal bars
        y_pos = np.arange(len(llm_names))
        left = np.zeros(len(llm_names))
        
        bars = []
        for baseline in baseline_types:
            color = self.config['default_colors'].get(baseline, '#777777')
            bar = ax.barh(y_pos, bias_data[baseline], left=left, color=color)
            bars.append(bar)
            left += bias_data[baseline]
        
        # Add labels and annotations
        for i, baseline in enumerate(baseline_types):
            for j, bar in enumerate(bars[i]):
                width = bar.get_width()
                if width > 5:  # Only add text if bar is wide enough
                    ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                            f"{bias_data[baseline][j]:.1f}%", 
                            ha='center', va='center', color='white', fontweight='bold')
        
        # Set axis labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(llm_names)
        ax.set_xlabel('Political Bias Percentage')
        
        title = kwargs.get('title', 'Political Bias Distribution of LLMs')
        ax.set_title(title)
        
        # Add legend
        legend_handles = []
        legend_labels = []
        
        for baseline in baseline_types:
            color = self.config['default_colors'].get(baseline, '#777777')
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            legend_labels.append(baseline.capitalize())
        
        ax.legend(legend_handles, legend_labels, loc='lower right')
        
        # Improve layout
        fig.tight_layout()
        
        # Show figure
        if kwargs.get('show', True):
            plt.show()
            
        return fig

    def visualize_llm_baseline_comparison(self, **kwargs):
        """
        Create subplot visualization showing each LLM's similarity to all baselines
        
        Args:
            **kwargs: Additional options including:
                - figsize: Figure size as (width, height) tuple
                - title: Plot title
                - show: Whether to display the plot (default: True)
                - llm_filter: List of LLM names to include (default: all)
                - use_combined: Whether to use combined documents for LLMs with multiple docs (default: True)
                - highlight_deviation: Whether to highlight deviation from mean (default: True)
                
        Returns:
            matplotlib Figure object
        """
        # Calculate positions for all documents
        positions = self.calculate_document_positions()
        if not positions:
            print("No document positions to visualize")
            return None
            
        # Get baseline types
        baseline_types = list(self.baselines.keys())
        if not baseline_types:
            print("No baseline types defined")
            return None
            
        # Get LLM document labels based on filter and combined preference
        use_combined = kwargs.get('use_combined', True)
        llm_filter = kwargs.get('llm_filter')
        
        llm_docs = {}
        if 'metadata' in self.data:
            for label in positions.keys():
                if label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    
                    # Filter by type and group
                    if metadata.get('type') == 'llm':
                        group = metadata.get('group')
                        
                        # Filter by LLM name if specified
                        if llm_filter is not None and group not in llm_filter:
                            continue
                            
                        # Prefer combined documents if available
                        if use_combined and metadata.get('is_group', False):
                            llm_docs[group] = label
                        elif not use_combined and not metadata.get('is_group', False):
                            # For non-combined, use first document of each LLM
                            if group not in llm_docs:
                                llm_docs[group] = label
        
        # Calculate mean similarity for each baseline
        baseline_means = {baseline: 0 for baseline in baseline_types}
        for label, pos in positions.items():
            if 'metadata' in self.data and label in self.data['metadata']:
                if self.data['metadata'][label].get('type') == 'llm':
                    for baseline in baseline_types:
                        if baseline in pos:
                            baseline_means[baseline] += pos[baseline]
        
        # Divide by number of LLM documents to get means
        num_llms = len([label for label in positions if
                        'metadata' in self.data and 
                        label in self.data['metadata'] and
                        self.data['metadata'][label].get('type') == 'llm'])
        
        if num_llms > 0:
            for baseline in baseline_means:
                baseline_means[baseline] /= num_llms
        
        # Set up figure
        n_llms = len(llm_docs)
        n_cols = min(3, n_llms)
        n_rows = (n_llms + n_cols - 1) // n_cols
        
        figsize = kwargs.get('figsize', (15, 5 * n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single row/column cases
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Create bar charts for each LLM
        for i, (llm_name, label) in enumerate(llm_docs.items()):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            
            if label in positions:
                # Get similarity values for this LLM
                pos = positions[label]
                
                # Extract values for each baseline
                baseline_values = []
                baseline_labels = []
                bar_colors = []
                
                for baseline in baseline_types:
                    if baseline in pos:
                        baseline_values.append(pos[baseline])
                        baseline_labels.append(baseline.capitalize())
                        
                        # Set color based on deviation from mean if requested
                        if kwargs.get('highlight_deviation', True) and baseline in baseline_means:
                            mean = baseline_means[baseline]
                            deviation = pos[baseline] - mean
                            
                            if deviation > 0.05:  # Above average
                                color = 'green'
                            elif deviation < -0.05:  # Below average
                                color = 'red'
                            else:  # Around average
                                color = 'blue'
                        else:
                            color = self.config['default_colors'].get(baseline, '#777777')
                            
                        bar_colors.append(color)
                    
                # Create bar chart
                x = np.arange(len(baseline_values))
                bars = ax.bar(x, baseline_values, color=bar_colors)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
                
                # Set labels and title
                ax.set_xticks(x)
                ax.set_xticklabels(baseline_labels, rotation=45, ha='right')
                
                # Get display name for LLM
                display_name = llm_name.upper()
                ax.set_title(display_name)
                
                # Adjust y-axis to start from 0 and have consistent scale
                ax.set_ylim(0, 1)
                
                # Add mean line for each baseline if requested
                if kwargs.get('highlight_deviation', True):
                    for j, baseline in enumerate(baseline_types):
                        if baseline in baseline_means:
                            ax.axhline(y=baseline_means[baseline], color='gray', 
                                       linestyle='--', alpha=0.5, xmin=j/len(baseline_types), 
                                       xmax=(j+1)/len(baseline_types))
        
        # Hide empty subplots
        for i in range(n_llms, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        # Set overall title
        title = kwargs.get('title', 'LLM Similarity to Political Baselines')
        fig.suptitle(title, fontsize=16)
        
        # Improve layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Show figure
        if kwargs.get('show', True):
            plt.show()
            
        return fig

    def visualize_3d_features(self, **kwargs):
        """Create a 3D scatter plot comparing documents on three features"""
            
        # Get document labels
        doc_labels = kwargs.get('document_labels', list(self.documents.keys()))
        
        # Define features to compare (need exactly 3 for 3D plot)
        default_features = ['numwords', 'lexical_diversity', 'avg_sentence_length']
        features = kwargs.get('features', default_features)
        
        # Ensure we have exactly 3 features
        if len(features) != 3:
            print(f"Warning: 3D scatter plot requires exactly 3 features. Got {len(features)}. Using first 3 or defaults.")
            if len(features) > 3:
                features = features[:3]
            else:
                features = default_features[:len(features)] + default_features[len(features):3]
        
        # Set up 3D scatter plot
        figsize = kwargs.get('figsize', (12, 10))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Collect data
        x_data = []
        y_data = []
        z_data = []
        valid_labels = []
        
        for label in doc_labels:
            # Check if all three features exist for this document
            valid = True
            feature_values = []
            
            for feature in features:
                if feature in self.data and label in self.data[feature]:
                    feature_values.append(self.data[feature][label])
                else:
                    valid = False
                    break
            
            if valid:
                x_data.append(feature_values[0])
                y_data.append(feature_values[1])
                z_data.append(feature_values[2])
                valid_labels.append(label)
        
        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        z_data = np.array(z_data)
        
        # Determine colors based on metadata grouping if specified
        colors = None
        group_by = kwargs.get('group_by_metadata')
        
        if group_by and 'metadata' in self.data:
            # Extract groups from metadata
            groups = []
            for label in valid_labels:
                if label in self.data['metadata'] and group_by in self.data['metadata'][label]:
                    groups.append(self.data['metadata'][label][group_by])
                else:
                    groups.append('unknown')
            
            # Create color map
            unique_groups = list(set(groups))
            color_map = {group: i for i, group in enumerate(unique_groups)}
            colors = [color_map[group] for group in groups]
            
            # Create legend handles
            from matplotlib.lines import Line2D
            cmap = plt.cm.tab10
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=cmap(color_map[group] % 10), 
                       markersize=10, label=group)
                for group in unique_groups
            ]
        
        # Plot the scatter points
        scatter = ax.scatter(x_data, y_data, z_data, c=colors, cmap='tab10', s=100, alpha=0.7)

        # Add labels for each point
        for i, label in enumerate(valid_labels):
            ax.text(x_data[i], y_data[i], z_data[i], label, size=8)

        # Set axis labels
        ax.set_xlabel(features[0].replace('_', ' ').title())
        ax.set_ylabel(features[1].replace('_', ' ').title())
        ax.set_zlabel(features[2].replace('_', ' ').title())

        # Add title
        title = kwargs.get('title', '3D Feature Comparison of Documents')
        ax.set_title(title)

        # Add legend if we have groups
        if colors is not None:
            # Create legend directly from the scatter plot
            unique_groups = list(set(groups))
            legend_handles = []
            legend_labels = []
            
            for group in unique_groups:
                group_indices = [i for i, g in enumerate(groups) if g == group]
                if group_indices:
                    # Use the first point of each group for the legend
                    idx = group_indices[0]
                    legend_handles.append(
                        plt.Line2D([0], [0], 
                                  marker='o', 
                                  color='w',
                                  markerfacecolor=plt.cm.tab10(color_map[group] % 10), 
                                  markersize=10)
                    )
                    legend_labels.append(group)
            
            ax.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(1.1, 1.0))

        # Add grid
        ax.grid(True)
        
        # Show figure
        if kwargs.get('show', True):
            plt.tight_layout()
            plt.show()
            
        return fig

    def visualize_text_features(self, **kwargs):
        """Create a visualization showing each document's features"""
            
        # Get document labels - default to all non-baseline documents
        doc_labels = kwargs.get('document_labels')
        if doc_labels is None:
                baseline_labels = set(self.baselines.values())
                doc_labels = [label for label in self.documents if label not in baseline_labels]
            
        # Get baseline types for similarity visualization
        baseline_types = kwargs.get('baseline_types')
        show_similarities = kwargs.get('show_similarities', True)
        
        # Calculate similarities only once if needed
        positions = None
        if show_similarities:
            if baseline_types is None:
                baseline_types = list(self.baselines.keys())
                
            # Calculate similarities to baselines
            positions = self.calculate_document_positions(baseline_types)
            
            # Set features to baseline similarities
            features = baseline_types
            title = kwargs.get('title', 'Document Similarity to Baselines')
        else:
            # Define features to display
            default_features = ['numwords', 'unique_words', 'avg_word_length', 'lexical_diversity']
            features = kwargs.get('features', default_features)
            title = kwargs.get('title', 'Document Text Features')
        
        # Set up figure
        n_docs = len(doc_labels)
        n_cols = min(3, n_docs)
        n_rows = (n_docs + n_cols - 1) // n_cols
        
        figsize = kwargs.get('figsize', (15, 5 * n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single row/column cases
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        # Get max values for consistent scaling
        max_values = {}
        for feature in features:
            if show_similarities and positions:
                # For similarities, max is always 1.0
                max_values[feature] = 1.0
            else:
                # For other features, find max across all documents
                if feature in self.data:
                    max_values[feature] = max(self.data[feature].get(label, 0) for label in doc_labels)
                else:
                    max_values[feature] = 1.0
            
        # Create bar charts for each document
        for i, label in enumerate(doc_labels):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            
            if show_similarities and label in positions:
                # Get similarity values for this document
                feature_values = [positions[label].get(feature, 0) for feature in features]
                feature_labels = [feature.replace('_', ' ').title() for feature in features]
                
                # Set colors based on baseline
                bar_colors = [self.config['default_colors'].get(feature, '#777777') for feature in features]
            else:
                # Get document features
                doc_features = {}
                for k, v in self.data.items():
                    if label in v:
                        doc_features[k] = v[label]
                
                # Extract values for selected features
                feature_values = []
                feature_labels = []
                bar_colors = []
                
                for feature in features:
                    if feature in doc_features:
                        feature_values.append(doc_features[feature])
                        feature_labels.append(feature.replace('_', ' ').title())
                        bar_colors.append('#1f77b4')  # Default blue
                    
            # Create bar chart
            x = np.arange(len(feature_values))
            bars = ax.bar(x, feature_values, width=0.6, color=bar_colors)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            
            # Set labels and title
            ax.set_xticks(x)
            ax.set_xticklabels(feature_labels, rotation=45, ha='right')
            
            # Get display name
            if 'metadata' in self.data and label in self.data['metadata']:
                metadata = self.data['metadata'][label]
                if 'model' in metadata and 'version' in metadata:
                    display_name = f"{metadata['model']} ({metadata['version']})"
                else:
                    display_name = label
            else:
                display_name = label
                
            ax.set_title(display_name)
            
            # Set consistent y-axis scale
            if show_similarities:
                ax.set_ylim(0, 1.0)
            else:
                # Get max values for visible features
                visible_features = [f for f, v in zip(features, feature_values) if v > 0]
                if visible_features:
                    max_y = max(max_values.get(f, 1.0) for f in visible_features) * 1.1
                    ax.set_ylim(0, max_y)
                else:
                    ax.set_ylim(0, 1.0)
            
        # Hide empty subplots
        for i in range(n_docs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
            
        # Set overall title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Show figure
        if kwargs.get('show', True):
            plt.show()
            
        return fig

    def visualize_comparative_features(self, **kwargs):
        """Create a radar chart comparing documents"""
            
        # Get document labels
        doc_labels = kwargs.get('document_labels', list(self.documents.keys()))
        
        # Define features to compare
        default_features = [
            'numwords', 'unique_words', 'avg_word_length', 'lexical_diversity', 'avg_sentence_length'
        ]
        features = kwargs.get('features', default_features)
        
        # Filter document labels by group if specified
        group_filter = kwargs.get('group_filter')
        if group_filter and 'metadata' in self.data:
            filtered_labels = []
            for label in doc_labels:
                if label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    if 'group' in metadata and metadata['group'] in group_filter:
                        filtered_labels.append(label)
            doc_labels = filtered_labels
        
        # Use group representatives if requested (e.g., combined documents)
        use_group_reps = kwargs.get('use_group_reps', False)
        if use_group_reps:
            group_docs = {}
            for label in doc_labels:
                if 'metadata' in self.data and label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    if 'group' in metadata:
                        group = metadata['group']
                        is_group_doc = metadata.get('is_group', False)
                        
                        # Prefer group documents, otherwise use first document of each group
                        if is_group_doc or group not in group_docs:
                            group_docs[group] = label
            
            # Replace document labels with group representatives
            doc_labels = list(group_docs.values())
        
        # Set up radar chart
        figsize = kwargs.get('figsize', (12, 8))
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Collect data more efficiently
        data = []
        labels = []
        
        for label in doc_labels:
            # Extract feature values directly
            feature_values = []
            valid = True
            
            for feature in features:
                if feature in self.data and label in self.data[feature]:
                    feature_values.append(self.data[feature][label])
                else:
                    valid = False
                    break
            
            if valid:
                data.append(feature_values)
                
                # Get display name
                if 'metadata' in self.data and label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    if 'model' in metadata:
                        display_name = metadata['model'].upper()
                    elif 'group' in metadata:
                        display_name = metadata['group'].upper()
                    else:
                        display_name = label
                else:
                    display_name = label
                
                labels.append(display_name)
            
        # Convert to numpy array and normalize
        data = np.array(data)
        
        if len(data) == 0:
            print("No valid documents to visualize")
            plt.close(fig)
            return None
            
        if kwargs.get('normalize', True):
            # Create normalizer for each feature
            for i in range(data.shape[1]):
                max_val = np.max(data[:, i])
                if max_val > 0:
                    data[:, i] = data[:, i] / max_val
            
        # Set up angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        data = np.column_stack((data, data[:, 0]))  # Close the data loop
        
        # Feature labels
        feature_labels = [f.replace('_', ' ').title() for f in features]
        
        # Choose colors
        cmap = cm.get_cmap('tab10', len(labels))
        
        # Plot each document
        for i, label in enumerate(labels):
            color = cmap(i)
            ax.plot(angles, data[i], linewidth=2, linestyle='solid', label=label, color=color)
            ax.fill(angles, data[i], alpha=0.1, color=color)
            
        # Configure axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels)
        
        ax.set_rlabel_position(0)
        if kwargs.get('normalize', True):
            ax.set_yticks([0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        
        # Add title and legend
        title = kwargs.get('title', 'Comparative Document Features')
        ax.set_title(title)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Show figure
        if kwargs.get('show', True):
            plt.tight_layout()
            plt.show()
            
        return fig

    def wordcount_sankey(self, word_list=None, k=5, **kwargs):
        """Create a Sankey diagram showing word flows between documents"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("This visualization requires plotly. Install with: pip install plotly")
            return None
            
        # Get document labels
        doc_labels = kwargs.get('document_labels', list(self.documents.keys()))
        
        # Filter document labels by group if specified
        group_filter = kwargs.get('group_filter')
        if group_filter and 'metadata' in self.data:
            filtered_labels = []
            for label in doc_labels:
                if label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    if 'group' in metadata and metadata['group'] in group_filter:
                        filtered_labels.append(label)
            doc_labels = filtered_labels
        
        # Use group representatives if requested (e.g., combined documents)
        use_group_reps = kwargs.get('use_group_reps', False)
        if use_group_reps:
            group_docs = {}
            for label in doc_labels:
                if 'metadata' in self.data and label in self.data['metadata']:
                    metadata = self.data['metadata'][label]
                    if 'group' in metadata:
                        group = metadata['group']
                        is_group_doc = metadata.get('is_group', False)
                        
                        # Prefer group documents, otherwise use first document of each group
                        if is_group_doc or group not in group_docs:
                            group_docs[group] = label
            
            # Replace document labels with group representatives
            doc_labels = list(group_docs.values())
        
        # Get word counts directly from the data structure
        exclude_words = kwargs.get('exclude_words', [])
        word_counts = {}
        
        # Only process documents that have word counts
        if 'wordcount' in self.data:
            for label in doc_labels:
                if label in self.data['wordcount']:
                    # Get existing word counts and filter excluded words
                    counts = {word: count for word, count in self.data['wordcount'][label].items() 
                             if word not in exclude_words}
                    word_counts[label] = counts
        
        # Build word list from words with total frequency >= k if not provided
        if word_list is None:
            # Calculate total frequency for each word across all documents
            total_word_counts = Counter()
            for label, counts in word_counts.items():
                total_word_counts.update(counts)
            
            # Filter words by total frequency across all documents
            word_list = [word for word, count in total_word_counts.most_common() if count >= k]
            
            # Limit to top N words if specified
            max_words = kwargs.get('max_words', 30)
            if len(word_list) > max_words:
                word_list = word_list[:max_words]
        
        # Create Sankey data
        sources = []
        targets = []
        values = []
        labels = []
        
        # Prepare display names for documents
        display_names = []
        for label in doc_labels:
            if 'metadata' in self.data and label in self.data['metadata']:
                metadata = self.data['metadata'][label]
                if 'model' in metadata:
                    display_name = metadata['model'].upper()
                elif 'group' in metadata:
                    display_name = metadata['group'].upper()
                else:
                    display_name = label
            else:
                display_name = label
            display_names.append(display_name)
        
        # Add document and word nodes
        doc_indices = {label: i for i, label in enumerate(doc_labels)}
        word_indices = {word: i + len(doc_labels) for i, word in enumerate(word_list)}
        
        labels = display_names + word_list
        
        # Add links
        for doc_label, counts in word_counts.items():
            doc_idx = doc_indices[doc_label]
            
            for word in word_list:
                if word in counts:
                    word_idx = word_indices[word]
                    count = counts[word]
                    
                    sources.append(doc_idx)
                    targets.append(word_idx)
                    values.append(count)
        
        # Set node colors
        node_colors = []
        highlight_baselines = kwargs.get('highlight_baselines', False)
        
        if highlight_baselines:
            # Define default colors for baseline types
            baseline_colors = kwargs.get('baseline_colors', {})
            
            for i, label in enumerate(labels):
                if i < len(doc_labels):  # Document nodes
                    doc_label = doc_labels[i]
                    
                    if doc_label in self.baselines.values():
                        # Find baseline type
                        baseline_type = None
                        for btype, blabel in self.baselines.items():
                            if blabel == doc_label:
                                baseline_type = btype
                                break
                        
                        # Set color based on baseline type
                        color = baseline_colors.get(baseline_type, 'rgba(128, 128, 128, 0.5)')
                        node_colors.append(color)
                    else:
                        # Check if it's an LLM document
                        if 'metadata' in self.data and doc_label in self.data['metadata']:
                            if self.data['metadata'][doc_label].get('type') == 'llm':
                                node_colors.append('rgba(44, 160, 44, 0.8)')  # Green for LLMs
                            else:
                                node_colors.append('rgba(128, 128, 128, 0.5)')  # Gray
                        else:
                            node_colors.append('rgba(128, 128, 128, 0.5)')  # Gray
                else:
                    # Word nodes
                    node_colors.append('rgba(255, 127, 14, 0.8)')  # Orange for words
        else:
            # Default colors - documents blue, words orange
            for i in range(len(labels)):
                if i < len(doc_labels):
                    node_colors.append('rgba(31, 119, 180, 0.8)')  # Blue for documents
                else:
                    node_colors.append('rgba(255, 127, 14, 0.8)')  # Orange for words
        
        # Link colors follow source node colors
        link_colors = [node_colors[source] for source in sources]
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])
        
        # Configure layout
        width = kwargs.get('width', 1200)
        height = kwargs.get('height', 800)
        title = kwargs.get('title', 'Word Distribution Across Documents')
        
        fig.update_layout(
            title_text=title,
            font_size=12,
            width=width,
            height=height
        )
        
        # Show figure
        if kwargs.get('show', True):
            renderer = kwargs.get('renderer', 'browser')
            fig.show(renderer=renderer)
        
        return fig