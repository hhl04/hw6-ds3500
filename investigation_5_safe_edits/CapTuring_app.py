"""
File: CapTuring_app.py

Description: Application that uses the CapTuring class to analyze documents
across different topics on a political spectrum with enhanced data handling
and visualization capabilities.
"""

from CapTuring import CapTuring
import os
import matplotlib.pyplot as plt

# *** CHANGE ONLY THIS LINE TO SWITCH TOPICS ***
TOPIC = "roe_vs_wade"  # Options: "ccp", "roe_vs_wade", "tariffs", etc.

def discover_human_perspectives(topic_path):
    """Discover available human perspectives for the topic
    
    Args:
        topic_path: Path to the topic directory
        
    Returns:
        dict: Dictionary mapping perspective names to file paths
    """
    human_path = os.path.join(topic_path, "human")
    if not os.path.exists(human_path):
        print(f"Error: Human perspectives folder not found at {human_path}")
        return {}
        
    perspectives = {}
    for perspective in os.listdir(human_path):
        perspective_path = os.path.join(human_path, perspective)
        if os.path.isdir(perspective_path):
            perspective_files = [
                os.path.join(perspective_path, f) 
                for f in os.listdir(perspective_path)
                if os.path.isfile(os.path.join(perspective_path, f))
            ]
            
            if perspective_files:
                perspectives[perspective] = {
                    'files': perspective_files,
                    'group_id': f"human_{perspective}",
                    'file_count': len(perspective_files)
                }
    
    return perspectives


def discover_llm_sources(topic_path):
    """Discover available LLM sources for the topic with improved grouping
    
    Args:
        topic_path: Path to the topic directory
        
    Returns:
        dict: Dictionary mapping LLM names to file data with improved structure
    """
    llm_path = os.path.join(topic_path, "llm")
    if not os.path.exists(llm_path):
        print(f"Error: LLM sources folder not found at {llm_path}")
        return {}
        
    llm_sources = {}
    for llm in os.listdir(llm_path):
        llm_source_path = os.path.join(llm_path, llm)
        if os.path.isdir(llm_source_path):
            files = [
                os.path.join(llm_source_path, f) 
                for f in os.listdir(llm_source_path) 
                if os.path.isfile(os.path.join(llm_source_path, f))
            ]
            
            if files:
                # Store both combined group info and individual files
                llm_sources[llm] = {
                    'files': files,
                    'group_id': llm,
                    'file_count': len(files)
                }
    
    return llm_sources


def get_stop_words_path(topic):
    """Load stop words for the given topic
    
    Args:
        topic: Name of the topic
        
    Returns:
        str: Path to stop words file, or None if not found
    """
    # First choice: Topic's support folder
    topic_path = os.path.join("documents", topic)
    support_path = os.path.join(topic_path, "support")
    
    if os.path.exists(support_path):
        topic_stop_words = os.path.join(support_path, "stop_words.txt")
        if os.path.exists(topic_stop_words):
            print(f"Using topic-specific stop words from {topic_stop_words}")
            return topic_stop_words
    
    # Second choice: General stop words file
    current_dir_stop_words = "stop_words.txt"
    if os.path.exists(current_dir_stop_words):
        print(f"Using general stop words from {current_dir_stop_words}")
        return current_dir_stop_words
        
    # No stop words found
    print(f"Warning: No stop words file found for topic '{topic}'. Using default stop words.")
    return None


def load_human_documents(analyzer, human_perspectives, clean_text_options):
    """Load human perspective documents as baselines
    
    Args:
        analyzer: CapTuring instance
        human_perspectives: Dictionary of human perspectives
        clean_text_options: Text cleaning options
    """
    print("\nLoading human perspective documents...")
    
    for perspective, perspective_data in human_perspectives.items():
        # Use first file as baseline
        baseline_file = perspective_data['files'][0]
        baseline_name = f"human_{perspective}"
        print(f"  - Loading {perspective} baseline: {os.path.basename(baseline_file)}")
        analyzer.load_text(
            source=baseline_file, 
            label=baseline_name,
            is_baseline=True, 
            baseline_type=perspective,
            metadata={
                "type": "baseline", 
                "perspective": perspective, 
                "group": baseline_name,
                "is_baseline": True
            },
            clean_text_options=clean_text_options
        )
        
        # Load additional files from this perspective
        additional_files = perspective_data['files'][1:]
        
        if additional_files:
            # Create a concatenated version for the group document
            concatenated_text = ""
            
            for i, file_path in enumerate(additional_files, 1):
                print(f"  - Loading additional {perspective} doc {i}: {os.path.basename(file_path)}")
                
                # Load file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Add to concatenated text
                concatenated_text += file_content + "\n\n"
                
                # Load individual document
                analyzer.load_text(
                    source=file_path,
                    label=f"{baseline_name}_{i}",
                    metadata={
                        "type": "human", 
                        "perspective": perspective, 
                        "group": baseline_name,
                        "is_individual": True
                    },
                    clean_text_options=clean_text_options
                )
            
            # Add concatenated version if there are multiple files
            if concatenated_text:
                analyzer.load_text(
                    source=concatenated_text,
                    label=f"{baseline_name}_combined",
                    metadata={
                        "type": "human", 
                        "perspective": perspective, 
                        "group": baseline_name,
                        "is_group": True,
                        "file_count": len(additional_files)
                    },
                    clean_text_options=clean_text_options
                )


def load_llm_documents(analyzer, llm_sources, clean_text_options):
    """Load LLM documents with group awareness
    
    Args:
        analyzer: CapTuring instance
        llm_sources: Dictionary of LLM sources
        clean_text_options: Text cleaning options
    """
    print("\nLoading LLM documents...")
    
    # Track concatenated text for each LLM group
    llm_concatenated = {}
    
    for llm_name, source_data in llm_sources.items():
        # Initialize concatenated text container
        llm_concatenated[llm_name] = ""
        
        # Load individual files
        for i, file_path in enumerate(source_data['files']):
            print(f"  - Loading {llm_name} doc {i+1}: {os.path.basename(file_path)}")
            
            # Load file content
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            # Add to concatenated version
            llm_concatenated[llm_name] += file_content + "\n\n"
            
            # Load individual document
            analyzer.load_text(
                source=file_path,
                label=f"{llm_name}_{i+1}",
                metadata={
                    "type": "llm", 
                    "model": llm_name, 
                    "version": i+1, 
                    "group": llm_name,
                    "is_individual": True
                },
                clean_text_options=clean_text_options
            )
        
        # Also load the concatenated version as a group document
        if source_data['file_count'] > 1:
            # Only add concatenated version if there are multiple files
            analyzer.load_text(
                source=llm_concatenated[llm_name],
                label=f"{llm_name}_combined",
                metadata={
                    "type": "llm", 
                    "model": llm_name, 
                    "version": "combined", 
                    "group": llm_name,
                    "is_group": True,
                    "file_count": source_data['file_count']
                },
                clean_text_options=clean_text_options
            )


def main():
    """Main function to run the NLP analysis"""
    # Get topic path
    topic_path = os.path.join("documents", TOPIC)
    
    # Load stop words and initialize analyzer
    stop_words_path = get_stop_words_path(TOPIC)
    
    analyzer = CapTuring()
    if not os.path.exists(topic_path):
        print(f"Error: Topic folder '{topic_path}' not found")
        return
    if stop_words_path:
        analyzer.load_stop_words(stop_words_path)
        analyzer.add_stop_words(stop_words_path)
    
    # Discover perspectives and LLMs
    human_perspectives = discover_human_perspectives(topic_path)
    llm_sources = discover_llm_sources(topic_path)
    
    if not human_perspectives:
        print(f"Error: No human perspectives found for topic '{TOPIC}'")
        return
        
    if not llm_sources:
        print(f"Error: No LLM sources found for topic '{TOPIC}'")
        return
    
    print(f"\nAnalyzing topic: {TOPIC}")
    print(f"Found {len(human_perspectives)} human perspectives: {', '.join(human_perspectives.keys())}")
    print(f"Found {len(llm_sources)} LLM sources: {', '.join(llm_sources.keys())}")
    
    # Define text cleaning options
    clean_text_options = {
        'remove_urls': True,
        'remove_html': True,
        'remove_punctuation': True,
        'remove_numbers': True,
        'remove_extra_whitespace': True
    }
    
    # Load human perspective documents as baselines
    load_human_documents(analyzer, human_perspectives, clean_text_options)
    
    # Load LLM documents with group awareness
    load_llm_documents(analyzer, llm_sources, clean_text_options)
    
    # Calculate similarities between all documents
    print("\nCalculating document similarities...")
    sim_matrix, labels = analyzer.calculate_similarity_matrix()
    
    # Print most similar documents for each baseline
    print("\nMost similar documents to each baseline:")
    for baseline_type, baseline_label in analyzer.get_baseline_documents().items():
        print(f"\n{baseline_type.upper()} baseline ({baseline_label}):")
        similar_docs = analyzer.get_most_similar_documents(baseline_label, n=3)
        for doc_label, similarity in similar_docs:
            print(f"  - {doc_label}: {similarity:.4f}")
    
    # Calculate document positions relative to baselines
    print("\nCalculating document positions on the political spectrum...")
    spectrum_positions = analyzer.calculate_document_positions()
    
    # Print summary for each LLM type
    print("\nSummary of LLM positions:")
    for llm_name, source_data in llm_sources.items():
        print(f"\n{llm_name.upper()} Analysis:")
        
        # Get the combined document label if it exists
        combined_label = f"{llm_name}_combined" if source_data['file_count'] > 1 else f"{llm_name}_1"
        
        if combined_label in spectrum_positions:
            positions = spectrum_positions[combined_label]
            
            # Print positions for each baseline
            for baseline, similarity in positions.items():
                print(f"  - {baseline}: {similarity:.4f}")
        else:
            print(f"  No position data found for {combined_label}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Word Count Sankey Diagram
    print("\n1. Word Count Sankey Diagram:")
    analyzer.wordcount_sankey(
        k=30,
        title=f"Word Distribution Across {TOPIC.upper()} Documents",
        show=True,
        highlight_baselines=True,
        baseline_colors={
            'left': 'rgba(31, 119, 180, 0.8)',      # Blue for left
            'right': 'rgba(214, 39, 40, 0.8)',      # Red for right
            'center': 'rgba(44, 160, 44, 0.8)',     # Green for center
        },
        use_group_reps=True
    )
    
    # 2. Similarity Heatmap
    print("\n2. Similarity Heatmap:")
    analyzer.visualize_similarity_heatmap(
        title=f"Political Bias Similarity Heatmap - {TOPIC.upper()}",
        show=True,
        use_clustering=True,
        annotate=True,
        use_combined=True,
        figsize=(10, 8)
    )
    
    # 3. Most Similar Baseline Classification
    print("\n3. Most Similar Baseline Classification:")
    analyzer.visualize_most_similar_baseline(
        title=f"Political Alignment of LLMs - {TOPIC.upper()}",
        show=True,
        use_combined=True,
        figsize=(12, 6)
    )
    
    # 4. Political Bias Horizontal Bars
    print("\n4. Political Bias Horizontal Bars:")
    analyzer.visualize_political_bias_bars(
        title=f"Political Bias Distribution - {TOPIC.upper()}",
        show=True,
        use_combined=True,
        figsize=(12, 6)
    )
    
    # 5. Per-Model Baselines Comparison
    print("\n5. Per-Model Baselines Comparison:")
    analyzer.visualize_llm_baseline_comparison(
        title=f"LLM Baseline Comparison - {TOPIC.upper()}",
        show=True,
        use_combined=True,
        highlight_deviation=True,
        figsize=(15, 10)
    )
    
    # 6. Document features subplots
    print("\n6. Document Features Subplots:")
    analyzer.visualize_text_features(
        title=f"Text Features by Document - {TOPIC.upper()}",
        show=True,
        show_similarities=False,
        figsize=(15, 12)
    )
    
    # 7. Comparative radar chart
    print("\n7. Comparative Document Features:")
    analyzer.visualize_comparative_features(
        title=f"Comparative Analysis of {TOPIC.upper()} Documents",
        show=True,
        normalize=True,
        features=['numwords', 'unique_words', 'avg_word_length', 'lexical_diversity', 'avg_sentence_length'],
        use_group_reps=True,
        figsize=(12, 8)
    )
    
    # 8. 3D Feature visualization
    print("\n8. 3D Feature Comparison:")
    analyzer.visualize_3d_features(
        title=f"3D Feature Comparison - {TOPIC.upper()}",
        show=True,
        features=['avg_word_length', 'lexical_diversity', 'avg_sentence_length'],
        group_by_metadata='group',
        figsize=(12, 10)
    )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()