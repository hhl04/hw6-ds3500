with open('stop_words.txt', 'r', encoding='utf-8') as f:
    words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"Found {len(words)} words")
    print(f"First 5 words: {words[:5]}")