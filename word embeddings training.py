### Embeddings training ###

import os
from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence

corpus_dir = '/mnt/c/TopicModelling/word embedding/corpus_lemma/'

# Define parameter space to explore
parameter_space = {
    'dimensionality': [100, 300],  # Exploring different dimensionalities
    'window_size': [5, 10],  # Window sizes to explore
    'iterations': [5, 10, 20],  # Number of iterations to explore
}

models_to_train = {
    'Word2Vec_CBOW': {'sg': 0, 'hs': 0},
    'Word2Vec_SkipGram': {'sg': 1, 'hs': 0},
    'FastText_CBOW': {'sg': 0, 'hs': 0},
    'FastText_SkipGram': {'sg': 1, 'hs': 0},
}

# Iterate over parameter combinations
for dimensionality in parameter_space['dimensionality']:
    for window_size in parameter_space['window_size']:
        for iterations in parameter_space['iterations']:
            # Apply the settings for each subfolder in the corpus directory
            for subfolder in os.listdir(corpus_dir):
                subfolder_path = os.path.join(corpus_dir, subfolder)
                if os.path.isdir(subfolder_path):
                    metadata_sentences = []
                    for file_name in os.listdir(subfolder_path):
                        file_path = os.path.join(subfolder_path, file_name)
                        file_sentences = LineSentence(file_path)
                        metadata_sentences.extend(file_sentences)
                    
                    for model_name, model_params in models_to_train.items():
                        if model_name.startswith('Word2Vec'):
                            model = Word2Vec(vector_size=dimensionality, window=window_size, min_count=5, workers=4, epochs=iterations, **model_params)
                        elif model_name.startswith('FastText'):
                            model = FastText(vector_size=dimensionality, window=window_size, min_count=5, workers=4, epochs=iterations, **model_params)
                        
                        model.build_vocab(metadata_sentences)
                        model.train(metadata_sentences, total_examples=model.corpus_count, epochs=model.epochs, compute_loss=True)
                        
                        # Generate filenames that reflect the parameter settings
                        model_filename = f"{subfolder.lower()}_{model_name.lower()}_dim{dimensionality}_window{window_size}_iter{iterations}.model"
                        kv_filename = f"{subfolder.lower()}_{model_name.lower()}_dim{dimensionality}_window{window_size}_iter{iterations}.kv"
                        
                        # Paths where to save the model and KeyedVectors
                        model_path = os.path.join('/mnt/c/TopicModelling/word embedding/third round lemma', model_filename)
                        kv_path = os.path.join('/mnt/c/TopicModelling/word embedding/third round lemma', kv_filename)
                        
                        # Save the model and its KeyedVectors
                        model.save(model_path)
                        model.wv.save(kv_path)