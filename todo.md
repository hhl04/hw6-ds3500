- Rewrite to group LLMs (Huy)
- Visualizations:
- Sankey (Shourya)
- Similarity per model per baseline (Sean)
- Radar Plot - better colors & grouping (Shourya)
- the one that adds to 100% (Sean)
- similarity heatmap with baselines on one axis and grouped models on the other (Sean)
- Most similar baseline per grouping, along with similarity score
- PCA visualization 


I was prompting claude back and forth about analysis pipeline and visualizations. copy those above in.

eventualy I'll need to write a next steps page into prompt.

## Potential Extensions

1. **Model Evolution**: Track political bias changes across LLM versions
2. **Prompt Sensitivity**: Analyze how prompt wording affects political bias
3. **Cross-topic Analysis**: Compare political bias patterns across different topics
4. **Temporal Analysis**: Examine how LLM political bias evolves over time
5. **Sentiment Correlation**: Explore relationship between sentiment and political bias
6. **Fine-tuning Effects**: Measure impact of fine-tuning on political bias




Documentation and Framework Structure

The assignment emphasizes reusability and extensibility. While your code shows good OOP principles, explicit documentation for how to extend the framework (add parsers, new visualizations) would be beneficial.

Parser Extensibility

The assignment specifically mentions implementing "the ability to specify a custom domain-specific parser." Your current code has a parsers dictionary in the __init__ method, but the functionality to add custom parsers isn't fully developed.

State Variable Structure

The assignment suggests a specific dictionary-within-dictionary structure for the state variable. Your implementation uses defaultdict(dict) which is a good approach but should be documented to show how it meets this requirement.





You've identified exactly the correct areas which have not been implemented in the code yet. 

Please write a new BOTTOM section that concisely explains what's missing, both broadly and/or specifically.

 It would be nice to be able to edit one additional line in the app file, which will make the entire investigation go from grouped by folder to individual, per LLM file analysis. 


8. **Linguistic Complexity Comparison**
   - Because some political positions might systematically use simpler/complex language

9. **Certainty Language Analysis**
   - Because some political positions or LLM responses might use more uncertain language.

10. **Sentiment Trajectory**
    - Line charts showing emotional tone changes throughout the progression of each txt document
    - Overlaid comparison of sentiment patterns by political position

11. **Topic Distribution Comparison**
    - Automatically extracted topics using LDA with top terms labeled
    - Topic prevalence normalized as percentage of document content

12. **2D PCA Visualization**

13. **3D PCA Visualization**