import yake

def yake_keywords(text, top_n=10):
    # Initialize YAKE with custom parameters (can be tuned as needed)
    language = "en"  # Language of the text
    max_ngram_size = 3  # Maximum size of the n-grams
    deduplication_threshold = 0.9  # Deduplication threshold
    deduplication_algo = 'seqm'  # Sequence matcher algorithm
    window_size = 1  # Window size
    num_keywords = top_n  # Number of keywords to extract

    kw_extractor = yake.KeywordExtractor(
        lan=language, 
        n=max_ngram_size, 
        dedupLim=deduplication_threshold, 
        dedupFunc=deduplication_algo, 
        windowsSize=window_size
    )

    # Extract keywords
    keywords = kw_extractor.extract_keywords(text)
    
    # Return only the top_n keywords
    top_keywords = [kw[0] for kw in keywords[:num_keywords]]
    return top_keywords






