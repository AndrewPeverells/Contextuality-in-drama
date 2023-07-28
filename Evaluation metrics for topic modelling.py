#topic proportion and distributions

import pandas as pd

# Load your topic modeling results into a pandas DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/AndrewPeverells/Topic-Modelling-for-Latin/main/doc_topics_full_chrono.csv')

# Calculate the average topic proportions for each topic
avg_topic_proportions = df.iloc[:, 1:].mean()

# Print the average topic proportions
print(avg_topic_proportions)

# Plot the average topic proportions
import matplotlib.pyplot as plt

plt.bar(avg_topic_proportions.index, avg_topic_proportions.values)
plt.xlabel('Topic')
plt.ylabel('Average Proportion')
plt.title('Average Topic Proportions')
plt.xticks(rotation=90)
plt.show()

# Analyze the distribution of topic proportions across different chunks
df['topic_with_highest_proportion'] = df.iloc[:, 1:].idxmax(axis=1)
topic_distribution = df['topic_with_highest_proportion'].value_counts()

# Print the topic distribution
print(topic_distribution)

# Plot the topic distribution
plt.bar(topic_distribution.index, topic_distribution.values)
plt.xlabel('Topic')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.title('Topic Distribution')
plt.show()

#nb: outliers are not to be trusted.shave off the highest and the lowest. topic 24 and 20 are both noise.

#scores based on the topic stability: Mean Similarity, Standard Deviation, Maximum Similarity, 
#Percentage of Zero Similarity Scores, Percentage of Non-zero Similarity Scores

import numpy as np

# Replace these with your actual results
similarity_scores = np.array([0.8, 0.9, 0.7, 0.6, 0.8, 0.9])
num_pairs = len(similarity_scores)

# Calculate mean similarity
mean_similarity = np.mean(similarity_scores)

# Calculate standard deviation of similarity
stddev_similarity = np.std(similarity_scores)

# Calculate maximum similarity difference
max_similarity_difference = np.max(similarity_scores) - np.min(similarity_scores)

# Calculate percentage of zero similarity scores
num_zero_similarity = len(similarity_scores[similarity_scores == 0])
percent_zero_similarity = (num_zero_similarity / num_pairs) * 100

# Calculate percentage of non-zero similarity scores
num_nonzero_similarity = len(similarity_scores[similarity_scores > 0])
percent_nonzero_similarity = (num_nonzero_similarity / num_pairs) * 100

# Print the results
print("Mean Similarity: ", mean_similarity)
print("Standard Deviation of Similarity: ", stddev_similarity)
print("Maximum Similarity Difference: ", max_similarity_difference)
print("Percentage of Zero Similarity Scores: ", percent_zero_similarity, "%")
print("Percentage of Non-zero Similarity Scores: ", percent_nonzero_similarity, "%")

#inter-annotator agreement for the evaluation: cohen's kappa

from sklearn.metrics import cohen_kappa_score

# Example dataframes with topic numbers and labels for annotators
df_annotator1 = pd.DataFrame({'topics': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                              'labels': ["christian religion", "noise", "noise", "kingship-military", "kingship-christian religion", "crime", "family", "grief", "feast", "kingship-religion", "Sodom", "christian religion", "love", "kingship", "education", "Joseph's dreams", "feast", "grief", " NaN", "crime", "noise", "family", "religion", "kingship - christian religion", "life/death - family ties?", "noise", "joy", "family", "feast", "Prodigal son"]})

df_annotator2 = pd.DataFrame({'topics': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                              'labels': ["christ", "noise", "noise", "warfare", "religion", "crime", "family", "violence", "food and wine", "divine power", "Sodom", "religion", "love", "kingship", "education", "Joseph's dreams", "feast", "grief", "first person", "crime", "noise", "family", "christ", "kingship", "life and death", "noise", "joy", "family", "feast", "Prodigal"]})

df_annotator3 = pd.DataFrame({'topics': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                              'labels': ["God & Christ", "noise", "noise", "Emperor/Caesar and arms ", "Emperor + Christ ", "Crime + sister ", "Prodigal", "Faith & heresy ", "Banquet", "God/Sun ruler of the world", " Sin & punishment", "Martyrdom, christ", "love", "King & Queen ", "noise", "Joseph's dreams", " Acolastus + interjections", "Samson & Delilah", "NaN", "(Divine) punishment", "noise", "Hermenigildus + battle between brothers", "Christ + Emperor", "kingship", "God, life & death ", "noise", "Love + its pleasures", "Joseph & his brothers ", "Hospitality", "Return of the Prodigal"]})

# Extract labels from dataframes
annotator1 = df_annotator1['labels']
annotator2 = df_annotator2['labels']
annotator3 = df_annotator3['labels']

# Calculate Cohen's Kappa
kappa_annotator1_annotator2 = cohen_kappa_score(annotator1, annotator2)
kappa_annotator1_annotator3 = cohen_kappa_score(annotator1, annotator3)
kappa_annotator2_annotator3 = cohen_kappa_score(annotator2, annotator3)

print("Cohen's Kappa (Annotator 1 vs Annotator 2):", kappa_annotator1_annotator2)
print("Cohen's Kappa (Annotator 1 vs Annotator 3):", kappa_annotator1_annotator3)
print("Cohen's Kappa (Annotator 2 vs Annotator 3):", kappa_annotator2_annotator3)


#perplexity scores

import pandas as pd
from gensim.models import Phrases
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import math

# Load the topic model from the CSV dataframe
df = pd.read_csv("/mnt/c/TopicModelling/results/evaluation/model.csv")

# Get the list of keywords for each topic
topic_keywords = df['keywords'].tolist()

# Tokenize the keywords into lists of words
topic_keywords_tokens = [word_tokenize(topic) for topic in topic_keywords]

# Create a Phrases model to generate n-grams
ngram_model = Phrases(topic_keywords_tokens, threshold=1, delimiter=' ')

# Define the list of n-gram values to calculate perplexity for
ngram_values = [2, 3, 4]

# Iterate through the n-gram values
for n in ngram_values:
    print(f"Perplexity for n-gram {n}:")
    for i, topic_keywords_token in enumerate(topic_keywords_tokens):
        # Generate n-grams from the list of words
        ngrams_list = list(ngrams(topic_keywords_token, n))
        
        # Convert n-grams back to string format
        ngrams_string_list = [' '.join(ngram) for ngram in ngrams_list]
        
        # Join n-grams using Phrases model delimiter
        ngrams_string = ' '.join(ngrams_string_list)
        
        # Tokenize n-grams string into list of words
        ngrams_tokens = word_tokenize(ngrams_string)
        
        # Create a Dictionary for the n-grams tokens
        ngrams_dictionary = Dictionary([ngrams_tokens])
        
        # Calculate the number of tokens in the n-grams dictionary
        ngrams_dictionary_size = len(ngrams_dictionary)
        
        # Create a bag-of-words representation of the n-grams tokens
        ngrams_bow = ngrams_dictionary.doc2bow(ngrams_tokens)
        
        # Train an LDA model on the n-grams tokens
        lda_model = LdaModel(corpus=[ngrams_bow], id2word=ngrams_dictionary, num_topics=1, iterations=1000)
        
        # Calculate the perplexity of the LDA model
        perplexity = lda_model.log_perplexity([ngrams_bow])
        
        # Calculate the exponent of the perplexity to get the actual perplexity value
        perplexity = math.exp(-perplexity)
        
        print(f"Topic {i}: {perplexity}")

#gensim's coherence and perplexity

import pandas as pd
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import nltk
from nltk.tokenize import word_tokenize

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/AndrewPeverells/Topic-Modelling-for-Latin/main/doc_topics_full_chrono.csv')

# Extract the relevant columns for topic modeling
texts = df['names'].tolist() # assuming 'text' is the column name containing the text chunks
topic_cols = df.columns[1:] # assuming topic columns are named as topic0, topic1, ..., topic29
doc_topics = df[topic_cols].values.tolist()

# Tokenize the text chunks
texts_tokenized = [word_tokenize(text) for text in texts]

# Create a Gensim dictionary
dictionary = Dictionary(texts_tokenized)

# Create a Gensim corpus
corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

# Train the LDA model
num_topics = len(topic_cols)
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

# Calculate perplexity
perplexity = lda_model.log_perplexity(corpus)
print("Perplexity:", perplexity)

# Calculate coherence using c_v coherence
coherence_model = CoherenceModel(model=lda_model, texts=texts_tokenized, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()
print("Coherence:", coherence_score)


#gensim's coherence: loop over the diagnostics file obtained with the mallet sub-routine code + visualisation

from sklearn.preprocessing import MinMaxScaler

# Initialize a dictionary to store the results
results = defaultdict(lambda: defaultdict(list))

# Initialize a list to store all coherence scores
all_coherence = []

# Loop over the directories
for topics in range(10, 101, 10):
    for iterations in range(100, 1001, 100):
        output_dir = f'output/{topics}_{iterations}'
        
        # Open the diagnostics file
        tree = ET.parse(os.path.join(output_dir, 'diagnostics.xml'))
        root = tree.getroot()

        # Extract the coherence for each topic and average them
        coherences = [float(topic.get('coherence')) for topic in root.iter('topic')]
        avg_coherence = np.mean(coherences)
        
        # Append to the results dictionary
        results[iterations]['topics'].append(topics)
        results[iterations]['coherence'].append(avg_coherence)

        # Append to the all coherence list
        all_coherence.append(avg_coherence)

# Scale coherence scores to 0-1 range
scaler = MinMaxScaler(feature_range=(0, 10))
all_coherence_scaled = scaler.fit_transform(np.array(all_coherence).reshape(-1, 1))

# Replace coherence scores in results dictionary with scaled coherence scores
i = 0
for iterations in results.keys():
    for j in range(len(results[iterations]['coherence'])):
        results[iterations]['coherence'][j] = all_coherence_scaled[i]
        i += 1

# Plot the coherence values
plt.figure(figsize=(20, 12))

for iterations, data in results.items():
    plt.plot(data['topics'], data['coherence'], marker='o', label=f'{iterations} iterations')

plt.xlabel('Number of Topics')
plt.ylabel('Average Scaled Coherence')
plt.title('Average Scaled Coherence Values for Different Numbers of Topics and Iterations')
plt.legend()
plt.grid(True)
plt.show()

#semantic overlap between the two sets of keyword-topic distributionsfor latin and italian: template

#round 2: semantic similarity with cosine and word2vec
#crime-grief cluster
#WikiWord2Vec model

from wikipedia2vec import Wikipedia2Vec
from numpy import dot
from numpy.linalg import norm

# Load pre-trained Word2Vec model
model_path = "/mnt/c/word2vec model/itwiki_20180420_300d.pkl"
model = Wikipedia2Vec.load(model_path)

# Calculate semantic similarity between two words using Word2Vec embeddings
def calculate_similarity(word1, word2):
    try:
        vec1 = model.get_word_vector(word1)
        vec2 = model.get_word_vector(word2)
        return dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    except KeyError:
        # Handle words not present in the vocabulary
        return 0

# Calculate semantic overlap between two sets of keywords
def calculate_semantic_overlap(set1, set2):
    similarities = []
    for word1 in set1:
        for word2 in set2:
            similarity = calculate_similarity(word1, word2)
            similarities.append(similarity)
    return sum(similarities) / len(similarities)

# Set 1 (Latin)
set1 = [
    'ohimè', 'fede', 'fede', 'delitto', 'dolore', 'lacrime', 'patria', 'sangue', 'impedimento',
    'furore', 'funerale', 'addio', 'vie', 'patria', 'eresia', 'fatica', 'fede', 'lacrime',
    'ohimè', 'sangue', 'viscere', 'arti', 'sporco', 'coro', 'felice', 'penitente', 'animo',
    'ferita', 'crudele', 'morte', 'grembo', 'luci', 'mano', 'crimine', 'sorella', 'crimine',
    'ragazza', 'principe', 'piangere', 'malattia', 'flebile', 'pena', 'innocente', 'lacrime',
    'fuggire', 'dannoso', 'peccato', 'furore', 'petto', 'crimine', 'peccato', 'dolore', 'testa',
    'animo', 'feroce', 'mano', 'luci', 'paura', 'dio', 'mente', 'orribile', 'timore', 'paura', 'ombra'
]

# Set 2 (Italian)
set2 = [
    'figlio', 'sangue', 'padre', 'morte', 'madre', 'cielo', 'pianto', 'ahi', 'ferro', 'figli',
    'pietà', 'fratello', 'vendetta', 'tiranno', 'figlia', 'morire', 'furore', 'petto', 'ohimè', 'innocente',
    'morte', 'vita', 'sole', 'cuore', 'dolore', 'cielo', 'fine', 'ohimè', 'crudele', 'morire',
    'pietà', 'occhi', 'dio', 'pena', 'pianto', 'infelice', 'anima', 'dolce', 'petto',
    'ohimè', 'morto', 'casa', 'capo', 'morta', 'testa', 'viva', 'povero', 'corpo', 'povera',
    'terra', 'porta', 'aiuto', 'traditore', 'mani', 'vecchio', 'morire', 'cuore', 'sangue',
    'cielo', 'dei', 'patria', 'seno', 'fatale', 'furore', 'preda', 'braccio', 'sacro', 'tiranni',
    'vendetta', 'tomba', 'fato', 'membra', 'mura', 'ombra', 'ohimè', 'aiuto', 'bosco',
    'mano', 'ohimè', 'correre', 'lasciare', 'prendere', 'inganno', 'soccorso', 'donne', 'vesti', 'fuggire', 'fuggita'
]

# Calculate semantic overlap
overlap_score = calculate_semantic_overlap(set1, set2)
print("Semantic overlap score:", overlap_score)
