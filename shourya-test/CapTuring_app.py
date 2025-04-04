""" File: CapTuring_app.py
Description: Application that uses the CapTuring class for NLP analysis
of documents across a political spectrum.
"""

import os
import matplotlib.pyplot as plt
from CapTuring import CapTuring

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
    
    # Load all documents
    for filename in document_files:
        filepath = os.path.join(documents_dir, filename)
        label = filename  # Use filename as label
        capturing.load_document(filepath, label)
    
    # Example: Set baseline documents (human articles representing different political views)
    # Assuming we have these files in the documents directory
    baseline_docs = []
    
    # Look for human conservative and progressive articles
    human_conservative = next((doc for doc in document_files if "human-conservative" in doc), None)
    human_progressive = next((doc for doc in document_files if "human-progressive" in doc), None)
    
    if human_conservative:
        baseline_docs.append(human_conservative)
    else:
        print("Warning: No human conservative article found. Expected filename containing 'human-conservative'.")
    
    if human_progressive:
        baseline_docs.append(human_progressive)
    else:
        print("Warning: No human progressive article found. Expected filename containing 'human-progressive'.")
    
    if baseline_docs:
        print(f"Using baseline documents: {baseline_docs}")
        capturing.set_baseline_documents(baseline_docs)
    else:
        print("Error: No baseline documents found. Cannot perform political spectrum analysis.")
        return
    
    # Compute TF-IDF and similarity matrix
    capturing.compute_similarity_matrix()
    
    # Analyze LLM articles by comparing to baseline human articles
    llm_docs = [doc for doc in document_files if "human" not in doc]
    print(f"\nAnalyzing {len(llm_docs)} LLM articles...")
    
    for doc in llm_docs:
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
    
    # 2. Political spectrum visualization
    fig2 = capturing.visualize_political_spectrum(llm_docs)
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