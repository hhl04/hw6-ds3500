"""
File: CapTuring_app.py

Description: Application file that uses the CapTuring class
to analyze documents across different topics and political spectrums.
"""

from CapTuring import CapTuring
import matplotlib.pyplot as plt
import os

# *** CHANGE ONLY THIS LINE TO SWITCH TOPICS ***
TOPIC = "ccp"  # Options: "ccp", "roe_vs_wade", etc.

def discover_human_perspectives(topic_path):
    """Discover available human perspectives for the given topic"""
    human_path = os.path.join(topic_path, "human")
    if not os.path.exists(human_path):
        print(f"Error: Human perspectives folder not found at {human_path}")
        return {}
        
    perspectives = {}
    for perspective in os.listdir(human_path):
        perspective_path = os.path.join(human_path, perspective)
        if os.path.isdir(perspective_path):
            # Get the first file in the perspective folder
            files = [f for f in os.listdir(perspective_path) if os.path.isfile(os.path.join(perspective_path, f))]
            if files:
                perspectives[perspective] = os.path.join(perspective_path, files[0])
    
    return perspectives

def discover_llm_sources(topic_path):
    """Discover available LLM sources and their documents for the given topic"""
    llm_path = os.path.join(topic_path, "llm")
    if not os.path.exists(llm_path):
        print(f"Error: LLM sources folder not found at {llm_path}")
        return {}
        
    llm_sources = {}
    for llm in os.listdir(llm_path):
        llm_source_path = os.path.join(llm_path, llm)
        if os.path.isdir(llm_source_path):
            # Get all files for this LLM source
            files = [os.path.join(llm_source_path, f) for f in os.listdir(llm_source_path) 
                    if os.path.isfile(os.path.join(llm_source_path, f))]
            if files:
                llm_sources[llm] = files
    
    return llm_sources

def main():
    """Main function to run the NLP analysis"""
    # Documents folder structure
    topic_path = os.path.join("documents", TOPIC)
    
    # Verify the topic folder exists
    if not os.path.exists(topic_path):
        print(f"Error: Topic folder '{topic_path}' not found")
        return
    
    # Check for topic-specific stop words in the support folder
    support_folder = os.path.join(topic_path, "support")
    stop_words_path = None
    
    if os.path.exists(support_folder):
        potential_stop_words = os.path.join(support_folder, "stop_words.txt")
        if os.path.exists(potential_stop_words):
            stop_words_path = potential_stop_words
            print(f"Found topic-specific stop words at {stop_words_path}")
    
    # Fallback to the older location if not found in support folder
    if not stop_words_path:
        fallback_path = f"stop_words_{TOPIC}.txt"
        if os.path.exists(fallback_path):
            stop_words_path = fallback_path
            print(f"Using fallback stop words at {stop_words_path}")
        else:
            print(f"No stop words found for topic '{TOPIC}'. Using default stop words.")
    
    # Initialize the analyzer with stop words if found
    if stop_words_path:
        analyzer = CapTuring(custom_stop_words=stop_words_path)
    else:
        analyzer = CapTuring()
    
    # Verify the topic folder exists
    if not os.path.exists(topic_path):
        print(f"Error: Topic folder '{topic_path}' not found")
        return
    
    # Discover human perspectives and LLM sources
    human_perspectives = discover_human_perspectives(topic_path)
    llm_sources = discover_llm_sources(topic_path)
    
    # Check if we have both human perspectives and LLM sources
    if not human_perspectives:
        print(f"Error: No human perspectives found for topic '{TOPIC}'")
        return
        
    if not llm_sources:
        print(f"Error: No LLM sources found for topic '{TOPIC}'")
        return
    
    print(f"\nAnalyzing topic: {TOPIC}")
    print(f"Found {len(human_perspectives)} human perspectives: {', '.join(human_perspectives.keys())}")
    print(f"Found {len(llm_sources)} LLM sources: {', '.join(llm_sources.keys())}")
    
    # Load human perspectives as baselines
    print("\nLoading human perspective documents as baselines...")
    for perspective, file_path in human_perspectives.items():
        file_name = os.path.basename(file_path)
        print(f"  - Loading {perspective} perspective: {file_name}")
        analyzer.load_document(
            file_path,
            label=f"human_{perspective}",
            is_baseline=True,
            baseline_type=perspective
        )
    
    # Load LLM documents
    print("\nLoading LLM documents...")
    for llm_name, file_paths in llm_sources.items():
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            print(f"  - Loading {llm_name} document {i+1}: {file_name}")
            analyzer.load_document(
                file_path,
                label=f"{llm_name}_{i+1}"
            )
    
    # Calculate similarities between all documents
    print("\nCalculating document similarities...")
    analyzer.calculate_similarity_matrix()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Create a similarity heatmap for all documents
    print("Creating similarity heatmap...")
    analyzer.visualize_similarity_heatmap()
    
    # Visualize documents on the political spectrum
    print("Visualizing political spectrum...")
    # Try to use right and left as primary axes if available, otherwise use the first two perspectives
    primary_axes = []
    if "right" in human_perspectives and "left" in human_perspectives:
        primary_axes = ["right", "left"]
    else:
        # Use the first two perspectives
        primary_axes = list(human_perspectives.keys())[:2]
        
    analyzer.visualize_political_spectrum(primary_axes=primary_axes)
    
    # Compare word frequencies with and without stop words
    print("Comparing word frequencies (excluding stop words)...")
    analyzer.compare_word_frequencies(n=15, exclude_stop_words=True)
    
    print("Comparing word frequencies (including stop words)...")
    analyzer.compare_word_frequencies(n=15, exclude_stop_words=False)
    
    # Print summary of analysis
    print("\nSummary of key findings:")
    print("------------------------")
    
    spectrum_positions = analyzer.calculate_political_spectrum()
    
    # Show summary for each LLM source
    for llm_name, _ in llm_sources.items():
        print(f"\n{llm_name.upper()} Analysis:")
        
        # Find all documents for this LLM
        llm_docs = [label for label in spectrum_positions.keys() if label.startswith(llm_name)]
        
        for doc_label in sorted(llm_docs):
            positions = spectrum_positions[doc_label]
            print(f"  - {doc_label}:")
            for baseline, similarity in positions.items():
                print(f"    {baseline}: {similarity:.4f}")
    
    # Print information about stop words used
    print("\nStop Words Information:")
    print(f"Total stop words used: {len(analyzer.get_stop_words())}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()