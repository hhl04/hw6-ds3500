""" File: CapTuring_app.py
Description: Application that uses the CapTuring class for NLP analysis
of documents across a political spectrum.
"""

import os
import matplotlib.pyplot as plt
from CapTuring import CapTuring
from collections import Counter

def main():
    """Main function to demonstrate the CapTuring framework"""
    # Create an instance of the CapTuring class
    capturing = CapTuring()
    
    # Example directory structure
    documents_dir = "documents"
    
    # Create documents directory if it doesn't exist
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created directory: {documents_dir}")
        print("Please add your document files to this directory before running analysis.")
        return
    
    # Check if documents exist
    document_files = [f for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f))]
    if not document_files:
        print(f"No documents found in {documents_dir}. Please add documents before running analysis.")
        return
    
    print(f"Found {len(document_files)} documents in {documents_dir}.")
    
    # Load all documents with appropriate categories
    for filename in document_files:
        filepath = os.path.join(documents_dir, filename)
        label = filename  # Use filename as label
        
        # Determine document category based on filename
        category = None
        if "ccp" in filename.lower():
            category = "CCP"
        elif "western" in filename.lower() or "human" in filename.lower():
            category = "Western"
            
        capturing.load_document(filepath, label, category=category)
        print(f"Loaded {label} as category: {category}")
    
    # Set baseline documents - using Wikipedia, American government, and Chinese government docs
    baseline_docs = []
    baseline_categories = []
    
    # Look for the baseline documents
    wikipedia_doc = next((doc for doc in document_files if "wikipedia" in doc.lower()), None)
    american_gov_doc = next((doc for doc in document_files if "american_government" in doc.lower()), None)
    chinese_gov_doc = next((doc for doc in document_files if "chinese_government" in doc.lower()), None)
    
    if wikipedia_doc:
        baseline_docs.append(wikipedia_doc)
        baseline_categories.append("Reference")
        print(f"Using Wikipedia document as baseline: {wikipedia_doc}")
    else:
        print("Warning: No Wikipedia document found. Expected filename containing 'wikipedia'.")
    
    if american_gov_doc:
        baseline_docs.append(american_gov_doc)
        baseline_categories.append("Western")
        print(f"Using American government document as baseline: {american_gov_doc}")
    else:
        print("Warning: No American government document found. Expected filename containing 'american_government'.")
    
    if chinese_gov_doc:
        baseline_docs.append(chinese_gov_doc)
        baseline_categories.append("CCP")
        print(f"Using Chinese government document as baseline: {chinese_gov_doc}")
    else:
        print("Warning: No Chinese government document found. Expected filename containing 'chinese_government'.")
    
    if baseline_docs:
        print(f"Using baseline documents: {baseline_docs}")
        capturing.set_baseline_documents(baseline_docs, categories=baseline_categories)
    else:
        print("Error: No baseline documents found. Cannot perform political spectrum analysis.")
        return
    
    # Compute TF-IDF and similarity matrix
    # Compute similarity matrix using cosine similarity on TF-IDF vectors
    capturing.compute_tfidf()
    capturing.compute_similarity_matrix()
    
    # Analyze LLM-generated CCP articles by comparing to baseline documents
    non_baseline_docs = [doc for doc in document_files if doc not in baseline_docs]
    print(f"\nAnalyzing {len(non_baseline_docs)} non-baseline articles...")
    
    # Group documents by category for analysis
    ccp_docs = [doc for doc in non_baseline_docs if "ccp" in doc.lower()]
    llm_docs = [doc for doc in non_baseline_docs if "ccp" not in doc.lower() and doc not in baseline_docs]
    
    print(f"Found {len(ccp_docs)} CCP documents and {len(llm_docs)} LLM documents")
    
    # Analyze all non-baseline documents
    for doc in non_baseline_docs:
        print(f"\nAnalysis for {doc}:")
        similarities = capturing.compare_to_baselines(doc)
        for baseline, score in similarities.items():
            print(f"  Similarity to {baseline}: {score:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Similarity heatmap
    fig1 = capturing.visualize_similarity_heatmap()
    if fig1:
        plt.figure(fig1.number)
        plt.savefig("similarity_heatmap.png")
        print("Saved similarity heatmap as 'similarity_heatmap.png'")
    
    # 2. Political spectrum visualization with custom labels
    spectrum_labels = None
    if len(baseline_docs) >= 2:
        # If we have American and Chinese government docs, use them as spectrum ends
        if american_gov_doc and chinese_gov_doc:
            spectrum_labels = ("American Government", "Chinese Government")
    
    fig2 = capturing.visualize_political_spectrum(
        non_baseline_docs, 
        spectrum_labels=spectrum_labels
    )
    if fig2:
        plt.figure(fig2.number)
        plt.savefig("political_spectrum.png")
        print("Saved political spectrum visualization as 'political_spectrum.png'")
    
    # 3. Word importance for each document
    for doc in document_files:
        fig3 = capturing.visualize_word_importance(doc)
        if fig3:
            plt.figure(fig3.number)
            filename = f"word_importance_{doc.replace('.', '_')}.png"
            plt.savefig(filename)
            print(f"Saved word importance visualization for {doc} as '{filename}'")
    
    # 4. Compare CCP vs Western document categories
    if ccp_docs and american_gov_doc:
        fig4 = capturing.compare_document_categories("CCP", "Western")
        if fig4:
            plt.figure(fig4.number)
            plt.savefig("ccp_vs_western_comparison.png")
            print("Saved CCP vs Western comparison as 'ccp_vs_western_comparison.png'")
    
    print("\nAnalysis complete!")

# Example of how to extend the framework with a custom parser
def custom_json_parser(filepath):
    """Example of a custom parser for JSON documents"""
    import json
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract text from JSON structure (adjust based on your JSON format)
        text = data.get('text', '')
        
        # Process text similar to default parser
        words = text.lower().split()
        
        results = {
            'text': text,
            'wordcount': Counter(words),
            'numwords': len(words),
            'unique_words': len(set(words)),
            'metadata': data.get('metadata', {})  # Store any metadata from JSON
        }
        
        return results
    except Exception as e:
        print(f"Error parsing JSON {filepath}: {e}")
        return {'text': '', 'wordcount': Counter(), 'numwords': 0, 'unique_words': 0}

# Example of how to use the custom parser
def example_with_custom_parser():
    capturing = CapTuring()
    # Example: Load a JSON document with custom parser
    # capturing.load_document('documents/example.json', parser=custom_json_parser)

if __name__ == "__main__":
    main()