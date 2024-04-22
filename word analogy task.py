# Word Analogy task

# Define word pairs for evaluation. Refer to the "religious concepts keywords.txt" file for reference.
word_pairs = [
    ('sanctus', 'pater'),
    ('gratia', 'fides'),
    ('christus', 'caesar')
    ('fides', 'ratio'),
    ('opus', 'peccatum')
    ...
]

analogy_tasks = [
    {'positive': ['caesar', 'christus'], 'negative': ['sanctus'], 'target': 'result'},
    {'positive': ['gratia', 'fides'], 'negative': ['pater'], 'target': 'similarity'},
    # Add more analogy tasks here
]

# Perform word analogy task for each model
for model_name, model in models.items():
    print(f"Word Analogy Task Results for model: {model_name}")
    for analogy_task in analogy_tasks:
        positive = analogy_task['positive']
        negative = analogy_task['negative']
        try:
            result = model.most_similar(positive=positive, negative=negative, topn=1)
            most_similar_word, similarity_score = result[0]
            print(f"{positive} - {negative} + {target} = {most_similar_word} (Similarity score: {similarity_score:.4f})")
        except KeyError as e:
            print(f"Error: {e}. One of the words not in vocabulary.")
    print()

# Compute semantic similarity for each model
for model_name, model in models.items():
    print(f"Semantic Similarity Results for model: {model_name}")
    for word_pair in word_pairs:
        word1, word2 = word_pair
        try:
            similarity = model.similarity(word1, word2)
            print(f"Semantic similarity between '{word1}' and '{word2}': {similarity:.4f}")
        except KeyError as e:
            print(f"Error: {e}. One of the words not in vocabulary.")
    print()