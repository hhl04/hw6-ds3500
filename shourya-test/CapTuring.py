""" File: CapTuring.py
Description: A reusable, extensible framework for comparative text analysis
designed to analyze documents across a political spectrum.
"""

from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

class CapTuring:
    def __init__(self):
        """
        Constructor for the CapTuring class.
        Initializes data structures to store document information and analysis results.
        """
        self.documents = {}  # Store raw text of documents
        self.data = defaultdict(dict)  # Store processed data and analysis results
        self.tfidf_matrix = None  # Store TF-IDF matrix for all documents
        self.vectorizer = None  # TF-IDF vectorizer
        self.similarity_matrix = None  # Store cosine similarity matrix
        self.baseline_docs = []  # List of baseline documents (e.g., human articles)
    
    def load_document(self, filepath, label=None, parser=None):
        """
        Load a document from a file and store its content.
        
        Args:
            filepath (str): Path to the document file
            label (str, optional): Label for the document. If None, uses filename
            parser (function, optional): Custom parser function. If None, uses default parser
        
        Returns:
            bool: True if document was loaded successfully, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found.")
            return False
        
        if label is None:
            label = os.path.basename(filepath)
        
        # Use default or custom parser
        if parser is None:
            results = self._default_parser(filepath)
        else:
            results = parser(filepath)
        
        # Store raw text
        self.documents[label] = results['text']
        
        # Store processed data
        for k, v in results.items():
            self.data[k][label] = v
            
        return True
    
    def _default_parser(self, filepath):
        """
        Default parser for text documents.
        
        Args:
            filepath (str): Path to the document file
            
        Returns:
            dict: Dictionary containing parsed data
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Basic text cleaning
            cleaned_text = self._clean_text(text)
            words = cleaned_text.split()
            
            results = {
                'text': text,  # Original text
                'cleaned_text': cleaned_text,  # Cleaned text
                'wordcount': Counter(words),  # Word frequency
                'numwords': len(words),  # Total word count
                'unique_words': len(set(words))  # Unique word count
            }
            
            print(f"Parsed: {filepath}")
            return results
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return {'text': '', 'wordcount': Counter(), 'numwords': 0, 'unique_words': 0}
    
    def _clean_text(self, text):
        """
        Clean text by removing special characters and converting to lowercase.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def set_baseline_documents(self, doc_labels):
        """
        Set baseline documents for comparison (e.g., human-written articles).
        
        Args:
            doc_labels (list): List of document labels to use as baselines
        """
        for label in doc_labels:
            if label in self.documents:
                self.baseline_docs.append(label)
            else:
                print(f"Warning: Baseline document '{label}' not found in loaded documents.")
    
    def compute_tfidf(self):
        """
        Compute TF-IDF vectors for all documents.
        
        Returns:
            numpy.ndarray: TF-IDF matrix
        """
        if not self.documents:
            print("Error: No documents loaded.")
            return None
        
        # Get document texts and labels in consistent order
        doc_labels = list(self.documents.keys())
        doc_texts = [self.documents[label] for label in doc_labels]
        
        # Compute TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(doc_texts)
        
        # Store document labels for reference
        self.data['doc_labels'] = doc_labels
        
        return self.tfidf_matrix
    
    def compute_similarity_matrix(self):
        """
        Compute cosine similarity matrix between all documents.
        
        Returns:
            numpy.ndarray: Similarity matrix
        """
        if self.tfidf_matrix is None:
            self.compute_tfidf()
            if self.tfidf_matrix is None:
                return None
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Store in data dictionary
        doc_labels = self.data['doc_labels']
        for i, label1 in enumerate(doc_labels):
            for j, label2 in enumerate(doc_labels):
                self.data['similarity'][(label1, label2)] = self.similarity_matrix[i, j]
        
        return self.similarity_matrix
    
    def get_similarity(self, doc1_label, doc2_label):
        """
        Get similarity between two documents.
        
        Args:
            doc1_label (str): Label of first document
            doc2_label (str): Label of second document
            
        Returns:
            float: Cosine similarity between documents
        """
        if 'similarity' not in self.data or (doc1_label, doc2_label) not in self.data['similarity']:
            self.compute_similarity_matrix()
        
        return self.data['similarity'].get((doc1_label, doc2_label), 0)
    
    def compare_to_baselines(self, doc_label):
        """
        Compare a document to all baseline documents.
        
        Args:
            doc_label (str): Label of document to compare
            
        Returns:
            dict: Dictionary mapping baseline labels to similarity scores
        """
        if not self.baseline_docs:
            print("Error: No baseline documents set.")
            return {}
        
        if doc_label not in self.documents:
            print(f"Error: Document '{doc_label}' not found.")
            return {}
        
        results = {}
        for baseline in self.baseline_docs:
            similarity = self.get_similarity(doc_label, baseline)
            results[baseline] = similarity
        
        return results
    
    def visualize_similarity_heatmap(self, figsize=(10, 8)):
        """
        Create a heatmap visualization of document similarities.
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
            if self.similarity_matrix is None:
                return None
        
        doc_labels = self.data['doc_labels']
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.similarity_matrix, cmap='YlOrRd')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va='bottom')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(doc_labels)))
        ax.set_yticks(np.arange(len(doc_labels)))
        ax.set_xticklabels(doc_labels)
        ax.set_yticklabels(doc_labels)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add text annotations
        for i in range(len(doc_labels)):
            for j in range(len(doc_labels)):
                text = ax.text(j, i, f"{self.similarity_matrix[i, j]:.2f}",
                              ha='center', va='center', color='black')
        
        ax.set_title('Document Similarity Matrix')
        fig.tight_layout()
        
        return fig
    
    def visualize_political_spectrum(self, doc_labels=None, figsize=(12, 6)):
        """
        Visualize documents on a political spectrum based on similarity to baseline documents.
        Assumes first baseline is 'left/progressive' and second is 'right/conservative'.
        
        Args:
            doc_labels (list, optional): List of document labels to visualize. If None, uses all non-baseline docs
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if len(self.baseline_docs) < 2:
            print("Error: Need at least two baseline documents (e.g., left and right).")
            return None
        
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
            if self.similarity_matrix is None:
                return None
        
        # If no doc_labels provided, use all non-baseline documents
        if doc_labels is None:
            doc_labels = [label for label in self.documents.keys() if label not in self.baseline_docs]
        
        # Get left and right baseline documents
        left_baseline = self.baseline_docs[0]
        right_baseline = self.baseline_docs[1]
        
        # Calculate political leaning score for each document
        # Higher positive values mean more right-leaning, negative values mean more left-leaning
        political_scores = {}
        for doc in doc_labels:
            right_sim = self.get_similarity(doc, right_baseline)
            left_sim = self.get_similarity(doc, left_baseline)
            # Calculate political score: positive is right-leaning, negative is left-leaning
            political_scores[doc] = right_sim - left_sim
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up the spectrum line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlim(-1, 1)
        
        # Plot each document on the spectrum
        y_positions = np.linspace(0.1, 0.9, len(political_scores))
        for i, (doc, score) in enumerate(political_scores.items()):
            ax.scatter(score, y_positions[i], s=100, label=doc)
            ax.annotate(doc, (score, y_positions[i]), xytext=(5, 0), 
                       textcoords='offset points', va='center')
        
        # Add labels for the spectrum ends
        ax.text(-0.95, -0.05, f"More similar to {left_baseline}\n(Progressive/Left)", 
                ha='left', va='top', fontsize=10)
        ax.text(0.95, -0.05, f"More similar to {right_baseline}\n(Conservative/Right)", 
                ha='right', va='top', fontsize=10)
        
        ax.set_title('Document Political Spectrum Analysis')
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    def get_top_terms(self, doc_label, n=10):
        """
        Get the top N most important terms for a document based on TF-IDF scores.
        
        Args:
            doc_label (str): Label of the document
            n (int): Number of top terms to return
            
        Returns:
            list: List of (term, score) tuples for top terms
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            self.compute_tfidf()
            if self.tfidf_matrix is None:
                return []
        
        # Get document index
        doc_labels = self.data['doc_labels']
        if doc_label not in doc_labels:
            print(f"Error: Document '{doc_label}' not found.")
            return []
        
        doc_idx = doc_labels.index(doc_label)
        
        # Get feature names and TF-IDF scores
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        tfidf_scores = self.tfidf_matrix[doc_idx].toarray().flatten()
        
        # Get indices of top terms
        top_indices = tfidf_scores.argsort()[-n:][::-1]
        
        # Return top terms and their scores
        return [(feature_names[i], tfidf_scores[i]) for i in top_indices]
    
    def visualize_word_importance(self, doc_label, n=10, figsize=(10, 6)):
        """
        Visualize the most important words in a document based on TF-IDF scores.
        
        Args:
            doc_label (str): Label of the document
            n (int): Number of top terms to visualize
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        top_terms = self.get_top_terms(doc_label, n)
        if not top_terms:
            return None
        
        # Extract terms and scores
        terms, scores = zip(*top_terms)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(terms))
        
        # Create horizontal bar chart
        ax.barh(y_pos, scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('TF-IDF Score')
        ax.set_title(f'Top {n} Important Terms in {doc_label}')
        
        fig.tight_layout()
        return fig