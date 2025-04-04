"""
File: textastic_app.py

Description: Main application that uses the Textastic class
to analyze CCP descriptions from different sources.
"""

from textastic import Textastic
import os


def main():
    # Initialize Textastic
    tt = Textastic()
    
    # Load stop words
    tt.load_stop_words('stop_words_ccp.txt')
    
    # Define file paths
    base_path = 'ccp_documents'
    
    # Define document groups
    reference_files = [
        ('wikipedia_ccp.txt', 'Wikipedia'),
        ('chinese_government_ccp.txt', 'Chinese_Govt'),
        ('american_government_ccp.txt', 'American_Govt')
    ]
    
    llm_files = [
        ('claude_1_ccp.txt', 'Claude_1'),
        ('claude_2_ccp.txt', 'Claude_2'),
        ('deepseek_1_ccp.txt', 'Deepseek_1'),
        ('deepseek_2_ccp.txt', 'Deepseek_2'),
        ('grok_1_ccp.txt', 'grok_1'),
        ('grok_2_ccp.txt', 'grok_2')
    ]
    
    # Load reference documents
    print("Loading reference documents...")
    for filename, label in reference_files:
        file_path = os.path.join(base_path, filename)
        tt.load_text(file_path, label, group='Reference')
    
    # Load LLM documents
    print("Loading LLM documents...")
    for filename, label in llm_files:
        file_path = os.path.join(base_path, filename)
        tt.load_text(file_path, label, group='LLM')
    
    print("Documents loaded successfully.")
    
    # 1. Basic word count comparison (after stop words removal)
    print("\nComparing document word counts (stop words removed)...")
    tt.compare_num_words()
    
    # 1b. Compare the effect of stop word removal
    print("\nAnalyzing stop word impact...")
    tt.analyze_stop_word_impact()
    
    # 2. Display similarities between LLMs and references
    print("\nCalculating similarities between LLMs and references...")
    similarity_matrix = tt.visualize_group_similarities(
        'LLM', 'Reference', 
        'Similarity Between LLM Outputs and Reference Texts'
    )
    
    # 3. Show which reference each LLM is most similar to
    print("\nFinding closest reference for each LLM output...")
    tt.plot_closest_references(
        'LLM', 'Reference',
        'Closest Reference Source for Each LLM Output'
    )
    
    # 4. Visualize bias distribution
    print("\nCalculating bias distribution...")
    tt.visualize_bias_distribution(
        'LLM', 'Reference',
        'Relative Bias of LLM Outputs'
    )
    
    # 5. Analyze common words
    print("\nAnalyzing common words across documents...")
    tt.analyze_common_words(n=15)
    
    # 6. Analyze political terms
    print("\nAnalyzing political term usage...")
    political_terms = [
        'communist', 'party', 'government', 'democracy', 'freedom',
        'control', 'authoritarian', 'regime', 'rights', 'censorship'
    ]
    tt.analyze_political_terms(
        political_terms,
        'Political Term Usage in CCP Descriptions'
    )
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()