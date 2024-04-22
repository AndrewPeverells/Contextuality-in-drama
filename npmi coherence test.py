# nPMI technique

from gensim.models import CoherenceModel

# Define a function to calculate nPMI coherence score for a given model
def calculate_npmi_coherence(model, dictionary, texts):
    # Extract topic-word distributions from the model
    topic_word_distributions = []
    for topic_id in range(model.num_topics):
        # Get the top words for the topic
        top_words = model.show_topic(topic_id)
        # Extract just the words (tokens)
        words = [word for word, _ in top_words]
        # Append the list of words to the topic_word_distributions
        topic_word_distributions.append(words)

    # Calculate nPMI coherence score
    coherence_model_npmi = CoherenceModel(
        topics=topic_word_distributions,
        texts=texts,
        dictionary=dictionary,
        coherence='c_npmi'  # Use 'c_npmi' instead of 'npmi'
    )
    coherence_npmi = coherence_model_npmi.get_coherence()

    return coherence_npmi

# Iterate over each model and calculate nPMI coherence score
for model_name, lda_model in lda_models.items():
    coherence_score = coherence_scores[model_name]
    npmicoherence = calculate_npmi_coherence(lda_model, dictionary, texts)
    print(f"Model: {model_name}, Coherence Score: {coherence_score}, nPMI Coherence Score: {npmicoherence}")