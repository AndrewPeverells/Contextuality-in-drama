#Second round on lemmatised texts

import os
from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence

# Set the path to your corpus directory
corpus_dir = '/mnt/c/TopicModelling/word embedding/corpus_lemma/'

# Define the models to train
models_to_train = {
    'Word2Vec_CBOW': {'sg': 0, 'hs': 0},
    'Word2Vec_SkipGram': {'sg': 1, 'hs': 0},
    'FastText_CBOW': {'sg': 0, 'hs': 0},
    'FastText_SkipGram': {'sg': 1, 'hs': 0},
}

# Iterate over each subfolder (metadata subdivision) in the corpus directory
for subfolder in os.listdir(corpus_dir):
    subfolder_path = os.path.join(corpus_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Initialize an empty list to store the tokenized sentences for the current metadata
        metadata_sentences = []

        # Iterate over each file in the subfolder
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)

            # Read the tokenized sentences from the file
            file_sentences = LineSentence(file_path)

            # Extend the list of sentences with the sentences from the current file
            metadata_sentences.extend(file_sentences)

        # Iterate over the models to train
        for model_name, model_params in models_to_train.items():
            # Train the Word2Vec or FastText model
            if model_name.startswith('Word2Vec'):
                model = Word2Vec(window=5, min_count=5, workers=16, **model_params)
            elif model_name.startswith('FastText'):
                model = FastText(window=5, min_count=5, workers=16, **model_params)

            # Build the vocabulary with the tokenized sentences for the current metadata
            model.build_vocab(metadata_sentences)

            # Train the model with the tokenized sentences for the current metadata
            model.train(metadata_sentences, total_examples=model.corpus_count, epochs=1, compute_loss=True)

            # Save the trained model for the current metadata and model type
            output_path = f'/mnt/c/TopicModelling/word embedding/second round lemma_{subfolder.lower()}_{model_name.lower()}.model'
            model.save(output_path)

#evaluation

from gensim.models import KeyedVectors

# Set the paths to your trained models
model_paths = {
    'cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_cbow.model',
    'cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_cbow.model',
    'skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_skipgram.model',
    'skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_skipgram.model',
    'fasttext_cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_cbow.model',
    'fasttext_cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_cbow.model',
    'fasttext_skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_skipgram.model',
    'fasttext_skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_skipgram.model'
}

# Load the models
models = {}
for model_name, model_path in model_paths.items():
    models[model_name] = KeyedVectors.load(model_path).wv

# Define word pairs for evaluation
word_pairs = [
    ('sanctus', 'pater'),
    ('gratia', 'fides'),
    ('christus', 'caesar')
    # Add more word pairs for evaluation
]

# Evaluate the models
for model_name, model in models.items():
    print(f'Evaluation results for model: {model_name}')
    print('Word Pair\t\tSimilarity Score')
    print('--------------------------------------')
    for word1, word2 in word_pairs:
        similarity_score = model.similarity(word1, word2)
        print(f'{word1} - {word2}\t\t{similarity_score:.4f}')
    print()


#word analogy task

from gensim.models import KeyedVectors

# Load the trained model paths
model_paths = {
    'cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_cbow.model',
    'cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_cbow.model',
    'skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_skipgram.model',
    'skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_skipgram.model',
    'fasttext_cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_cbow.model',
    'fasttext_cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_cbow.model',
    'fasttext_skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_skipgram.model',
    'fasttext_skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_skipgram.model'
}

# Load the models
models = {}
for model_name, model_path in model_paths.items():
    model = KeyedVectors.load(model_path).wv
    models[model_name] = model

# Perform word analogy for each model
for model_name, model in models.items():
    result = model.most_similar(positive=['christus', 'sanctus'], negative=['caesar'], topn=1)
    print(f"Word analogy result for {model_name}: {result}")

#word similarity task

#analyses on lemmatised models

from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import random

# Define the paths to the pre-trained models
model_paths = {
    'fasttext_cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_cbow.model',
    'fasttext_cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_cbow.model',
    }

# Load the word embedding models
models = {}
for metadata, model_path in model_paths.items():
    models[metadata] = KeyedVectors.load(model_path).wv

# Analyses
for metadata, model in models.items():
    print(f'--- Analysis for {metadata} metadata ---')

    # Word Similarity
    print(f'Words similar to "fides" in the {metadata} background:')
    similar_words = model.most_similar('scriptura', topn=30)
    for word, similarity in similar_words:
        print(word, similarity)
    print()
    
    # Word Analogies
    print(f'Word analogy test in the {metadata} background:')
    analogy_words = model.most_similar(positive=['scriptura', 'fides'], negative=['sanctus'], topn=30)
    for word, similarity in analogy_words:
        print(word, similarity)
    print()

#word similarity task: groups of words

#analysis for groups of words - one set of words for both backgrounds

from gensim.models import KeyedVectors

# Define the paths to the pre-trained models
model_paths = {
    'fasttext_cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_cbow.model',
    'fasttext_cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_cbow.model',
}

# Define the target words
target_words = ['pietas', 'iustitia', 'saluatio', 'salus']

# Load the word embedding models
models = {}
for metadata, model_path in model_paths.items():
    models[metadata] = KeyedVectors.load(model_path).wv

# Analyses
for metadata, model in models.items():
    print(f'--- Analysis for {metadata} words ---')
    similar_words = model.most_similar(target_words, topn=40)
    for word, similarity in similar_words:
        print(f'{word}: {similarity}')
    print()

#word-context analysis: n-grams

#word-context analysis (n-grams)

import os
import nltk
from nltk.util import ngrams

corpus_path = '/mnt/c/TopicModelling/word embedding/corpus_lemma/'
target_words = ["scriptura"]  # List of target words
context_window = 7  # Number of words before and after the target word

# Function to extract n-grams from a list of tokens
def extract_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Function to process a text file and track target word occurrences with n-gram contexts
def process_text_file(file_path, metadata):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        tokens = nltk.word_tokenize(text)
        ngram_tokens = extract_ngrams(tokens, context_window*2+1)
        
        for i in range(context_window, len(tokens)-context_window):
            if tokens[i] in target_words:
                context = " ".join(tokens[i-context_window:i] + [tokens[i]] + tokens[i+1:i+context_window+1])
                metadata.append((tokens[i], context))

# Process the corpus
metadata_catholic = []
metadata_protestant = []

# Process Catholic texts
catholic_folder = os.path.join(corpus_path, "catholic")
for filename in os.listdir(catholic_folder):
    file_path = os.path.join(catholic_folder, filename)
    process_text_file(file_path, metadata_catholic)

# Process Protestant texts
protestant_folder = os.path.join(corpus_path, "protestant")
for filename in os.listdir(protestant_folder):
    file_path = os.path.join(protestant_folder, filename)
    process_text_file(file_path, metadata_protestant)

# Print the results
print("Catholic metadata:")
for word, context in metadata_catholic:
    print(f"Word: {word}\tContext: {context}")

print("-----------------------------------------")  
    
print("Protestant metadata:")
for word, context in metadata_protestant:
    print(f"Word: {word}\tContext: {context}")


###############################################################################################################################

#external evaluation tasks: synonym-detection

#Performance task with SimLex-999 dataset

# Import required packages
import pandas as pd

# Path to the SimLex-999 dataset
simlex_path = "/mnt/c/TopicModelling/word embedding/SimLex-999/SimLex-999.txt"

# Load the dataset
simlex_df = pd.read_csv(simlex_path, sep="\t")

# Display the first few rows of the dataset
simlex_df.head()


#spearman's rank correlation coefficient
from scipy.stats import spearmanr
from gensim.models import KeyedVectors

# Define a function to evaluate a model using the SimLex-999 dataset
def evaluate_model_with_simlex(model_path, simlex_df):
    # Load the model
    model = KeyedVectors.load(model_path).wv
    
    # Initialize lists to store the model's similarity scores and the human-annotated similarity scores
    model_scores = []
    human_scores = []
    
    # Iterate over the rows in the SimLex-999 dataset
    for _, row in simlex_df.iterrows():
        word1 = row['word1']
        word2 = row['word2']
        
        # Check if both words are in the model's vocabulary
        if word1 in model.key_to_index and word2 in model.key_to_index:
            # Compute the similarity score for the word pair using the model
            model_score = model.similarity(word1, word2)
            model_scores.append(model_score)
            
            # Get the human-annotated similarity score for the word pair
            human_score = row['SimLex999']
            human_scores.append(human_score)
    
    # Compute the Spearman's rank correlation coefficient
    correlation, p_value = spearmanr(human_scores, model_scores)
    
    return correlation, p_value

# Define the paths to the trained models
model_paths = {
    'cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_cbow.model',
    'cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_cbow.model',
    'skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_skipgram.model',
    'skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_skipgram.model',
    'fasttext_cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_cbow.model',
    'fasttext_cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_cbow.model',
    'fasttext_skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_skipgram.model',
    'fasttext_skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_skipgram.model'
}

# Evaluate each model
for model_name, model_path in model_paths.items():
    correlation, p_value = evaluate_model_with_simlex(model_path, simlex_df)
    print(f"Evaluation results for model: {model_name}")
    print(f"Spearman's rank correlation coefficient: {correlation:.4f}")
    print(f"p-value: {p_value:.4f}")
    print()

#similarity with Sprugnoli et al. models: template

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

model_paths = {
    'cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_cbow.model',
    'cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_cbow.model',
    'skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_word2vec_skipgram.model',
    'skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_word2vec_skipgram.model',
    'fasttext_cbow_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_cbow.model',
    'fasttext_cbow_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_cbow.model',
    'fasttext_skipgram_protestant': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_protestant_fasttext_skipgram.model',
    'fasttext_skipgram_catholic': '/mnt/c/TopicModelling/word embedding/second round lemma/second round lemma_catholic_fasttext_skipgram.model'
}

# Load the provided model
provided_model_path = "/mnt/c/TopicModelling/word embedding/pre trained models/allLASLAlemmi-vector-100-nocase-w5-CBOW.vec"
provided_model = KeyedVectors.load_word2vec_format(provided_model_path)

# Evaluate each model
for model_name, model_path in model_paths.items():
    model = KeyedVectors.load(model_path)
    
    # Make sure the word 'rex' is in the vocabularies of both models
    if 'rex' in model.wv.key_to_index and 'rex' in provided_model.key_to_index:
        # Get the embeddings of the word 'rex' in both models
        embedding_model = model.wv['rex']
        embedding_provided_model = provided_model['rex']
        
        # Calculate the cosine similarity between the two embeddings
        similarity = cosine_similarity([embedding_model], [embedding_provided_model])
        
        print(f"Cosine similarity between the embeddings of the word 'rex' in {model_name} and the provided model: {similarity[0][0]}")
    else:
        print(f"The word 'rex' is not in the vocabulary of one or both models.")
    print()

#same task visualisation through boxplots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define a list of words for comparison
updated_words_to_compare = ['rex', 'pater', 'mater', 'fides', 'gratia']


# Define the paths to the models provided by Sprugnoli et al.
provided_model_paths = {
    'model1': '/mnt/c/TopicModelling/word embedding/pre trained models/allLASLA-lemmi-fast-100-CBOW-win10-min5.vec',
    'model2': '/mnt/c/TopicModelling/word embedding/pre trained models/allLASLA-lemmi-fast-100-SKIP-win5-min5.vec',
    'model3': '/mnt/c/TopicModelling/word embedding/pre trained models/allLASLAlemmi-vector-100-nocase-w10-SKIP.vec',
    'model4': '/mnt/c/TopicModelling/word embedding/pre trained models/allLASLAlemmi-vector-100-nocase-w5-CBOW.vec'
}

# Initialize a dictionary to store the results
similarity_results = {}

# Iterate over each of your models
for model_name, model_path in model_paths.items():
    
    # Load your model
    model = KeyedVectors.load(model_path)

    # Iterate over each of the provided models
    for provided_model_name, provided_model_path in provided_model_paths.items():

        # Load the provided model
        provided_model = KeyedVectors.load_word2vec_format(provided_model_path, binary=False)

        # Initialize a list to store the results for this comparison
        similarities = []
        
        # Iterate over each word
        for word in updated_words_to_compare:
            
            # Check if the word is in the vocabularies of both models
            if word in model.wv.key_to_index and word in provided_model.key_to_index:
                
                # Get the embeddings of the word in both models
                embedding_model = model.wv[word]
                embedding_provided_model = provided_model[word]
                
                # Calculate the cosine similarity between the two embeddings
                similarity = cosine_similarity([embedding_model], [embedding_provided_model])[0][0]
                
                # Store the result
                similarities.append(similarity)
        
        # Store the results for this comparison
        similarity_results[(model_name, provided_model_name)] = similarities

# Convert the results to a DataFrame for easier analysis and plotting
df = pd.DataFrame(similarity_results)

# Create a boxplot of the results
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.title("Distribution of Cosine Similarities for Each Model Comparison")
plt.xlabel("Model Comparison")
plt.ylabel("Cosine Similarity")
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.show()