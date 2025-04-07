"""
File: CapTuring_app.py

Description: Application that uses the CapTuring class to analyze documents
across different topics on a political spectrum.
"""

from CapTuring import CapTuring, StopWordsNotFoundError
import os

# *** CD to a parent folder i.e. investigation_3_new_modularized ***
# *** CHANGE ONLY THIS LINE TO SWITCH TOPICS ***
TOPIC = "ccp"  # Options: "ccp", "roe_vs_wade", "tariffs".


def load_stop_words(topic):
    """Load stop words for the given topic"""
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
    raise StopWordsNotFoundError(
        f"No stop words file found for topic '{topic}'. Please create either:\n"
        f"  - documents/{topic}/support/stop_words.txt (recommended)\n"
        f"  - stop_words.txt (in the current directory)"
    )


def discover_human_perspectives(topic_path):
    """Discover available human perspectives for the topic"""
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
                perspectives[perspective] = perspective_files
    
    return perspectives


def discover_llm_sources(topic_path):
    """Discover available LLM sources for the topic"""
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
                llm_sources[llm] = files
    
    return llm_sources


def main():
    """Main function to run the NLP analysis"""
    # Get topic path
    topic_path = os.path.join("documents", TOPIC)
    if not os.path.exists(topic_path):
        print(f"Error: Topic folder '{topic_path}' not found")
        return

    try:
        # Load stop words and initialize analyzer
        stop_words_path = load_stop_words(TOPIC)
        analyzer = CapTuring(custom_stop_words=stop_words_path, require_stop_words=True)
    except StopWordsNotFoundError as e:
        print(f"Error: {e}")
        return
    
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
    
    # Load human perspective documents as baselines
    print("\nLoading human perspective documents...")
    for perspective, file_paths in human_perspectives.items():
        # Use first file as baseline
        baseline_file = file_paths[0]
        baseline_name = f"human_{perspective}"
        print(f"  - Loading {perspective} baseline: {os.path.basename(baseline_file)}")
        analyzer.load_document(
            baseline_file,
            label=baseline_name,
            is_baseline=True,
            baseline_type=perspective,
            group=f"human_{perspective}"
        )
        
        # Load additional files from this perspective
        for i, file_path in enumerate(file_paths[1:], 1):
            print(f"  - Loading additional {perspective} doc {i}: {os.path.basename(file_path)}")
            analyzer.load_document(
                file_path,
                label=f"{baseline_name}_{i}",
                group=f"human_{perspective}"
            )
    
    # Load LLM documents
    print("\nLoading LLM documents...")
    for llm_name, file_paths in llm_sources.items():
        for i, file_path in enumerate(file_paths):
            print(f"  - Loading {llm_name} doc {i+1}: {os.path.basename(file_path)}")
            analyzer.load_document(
                file_path,
                label=f"{llm_name}_{i+1}",
                group=llm_name
            )
    
    # Calculate similarities
    print("\nCalculating document similarities...")
    analyzer.calculate_similarity_matrix()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_all_analyses()
    
    # Print analysis summary
    print("\nSummary of key findings:")
    print("------------------------")
    
    # Calculate group similarities to baselines
    group_similarities = analyzer.get_group_to_baseline_similarities()
    normalized_similarities = analyzer.normalize_baseline_similarities(group_similarities)
    
    # Show summary for each LLM source
    print("\nLLM Political Bias Analysis:")
    for llm_name in llm_sources.keys():
        if llm_name in normalized_similarities:
            print(f"\n  {llm_name.upper()} Political Distribution:")
            
            # Show percentages for each baseline
            for baseline, percentage in sorted(normalized_similarities[llm_name].items(), 
                                              key=lambda x: x[1], reverse=True):
                print(f"    {baseline}: {percentage:.1f}%")
    
    print(f"\nStop Words: {len(analyzer.get_stop_words())} words")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()