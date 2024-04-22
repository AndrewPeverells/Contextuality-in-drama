### Scikit-lda ###

import os
import itertools
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models.ldamodel import LdaModel

documents_list = []

for filename in os.listdir("/mnt/c/TopicModelling/corpus/lat/checked/chunked_full"):
    with open(os.path.join("/mnt/c/TopicModelling/corpus/lat/checked/chunked_full", filename), 'r') as f:
        text = f.read()
        documents_list.append(text)

# Preprocess documents for Gensim
texts = [[word for word in document.lower().split()] for document in documents_list]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Define parameter space
n_components_options = [5, 10, 15, 30, 40, 50, 100]  # Number of topics
learning_decay_options = [0.5, 0.7, 0.9]
use_tfidf_options = [True, False]  # Whether to use TF-IDF or not

results = []

for n_components, learning_decay, use_tfidf in itertools.product(n_components_options, learning_decay_options, use_tfidf_options):
    # Vectorize documents
    count_vect = CountVectorizer(lowercase=True, max_df=0.95, min_df=2, stop_words='english')
    x_counts = count_vect.fit_transform(documents_list)

    # Optionally apply TF-IDF
    if use_tfidf:
        tfidf_transformer = TfidfTransformer()
        x_transformed = tfidf_transformer.fit_transform(x_counts)
    else:
        x_transformed = x_counts

    # Convert to Gensim format
    corpus_gensim = Sparse2Corpus(x_transformed, documents_columns=False)

    # Train LDA Model
    lda = LdaModel(corpus=corpus_gensim, id2word=dictionary, num_topics=n_components, passes=10, random_state=0)

    # Calculate Coherence
    coherence_model = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    # Save LDA model
    model_filename = f"/mnt/c/TopicModelling/lda/models/lda_model_{n_components}_{learning_decay}_{use_tfidf}"
    lda.save(model_filename)

    # Save coherence score
    score_filename = f"coherence_score_{n_components}_{learning_decay}_{use_tfidf}.pkl"
    with open(score_filename, 'wb') as f:
        pickle.dump(coherence_score, f)

    results.append({
        'n_components': n_components,
        'learning_decay': learning_decay,
        'use_tfidf': use_tfidf,
        'coherence_score': coherence_score,
        'model_filename': model_filename,
        'score_filename': score_filename
    })

for result in results:
    print(f"Configuration: {result['n_components']} topics, decay {result['learning_decay']}, TF-IDF {result['use_tfidf']}, Coherence: {result['coherence_score']}")

# Save results to a df
    
import pandas as pd

df = pd.DataFrame(results_data)
print(df.head())
df.to_csv('/mnt/c/TopicModelling/lda_coherence_scores.csv', index=False)
df_loaded = pd.read_csv('/mnt/c/TopicModelling/lda_coherence_scores.csv')

# Visualisation

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(results)

df['configuration'] = df.apply(lambda x: f"{x['n_components']} topics, decay {x['learning_decay']}, TF-IDF {x['use_tfidf']}", axis=1)

# Sorting by coherence scores in ascending order
df.sort_values('coherence_score', inplace=True)

# Reset index for plotting
df.reset_index(drop=True, inplace=True)

# Creating a linear plot
plt.figure(figsize=(10, 6))
plt.plot(df['configuration'], df['coherence_score'], marker='o', linestyle='-')
plt.xticks(rotation=90)
plt.xlabel('Configuration')
plt.ylabel('Coherence Score')
plt.title('Coherence Scores Across Different LDA Model Configurations')
plt.ylim(0, 1)  # Scale the y-axis from 0 to 1
plt.grid(True)
plt.tight_layout()
plt.show()