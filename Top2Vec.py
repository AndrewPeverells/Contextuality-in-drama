### Top2Vec ###

from top2vec import Top2Vec
model = Top2Vec(documents_list, speed="deep-learn", workers = 16)

model.get_num_topics()

documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=1, num_docs=10)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()

topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_sizes)
print(topic_nums)

topic_words, word_scores, topic_nums = model.get_topics()

for words, scores, num in zip(topic_words, word_scores, topic_nums):
    print(num)
    print(f"Words: {words}")

#2

from top2vec import Top2Vec
model_min_20 = Top2Vec(documents_list, speed="deep-learn", workers = 16, min_count=20)

model_min_20.get_num_topics()

len(model_min_20.vocab)

documents, document_scores, document_ids = model_min_20.search_documents_by_topic(topic_num=1, num_docs=10)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()

topic_sizes, topic_nums = model_min_20.get_topic_sizes()
print(topic_sizes)
print(topic_nums)

topic_words, word_scores, topic_nums = model_min_20.get_topics()

for words, scores, num in zip(topic_words, word_scores, topic_nums):
    print(num)
    print(f"Words: {words}")


#bigrams

from top2vec import Top2Vec
model_min_20_ngram = Top2Vec(documents_list, speed="deep-learn", workers = 16, min_count=20, ngram_vocab=True)

model_min_20_ngram.get_num_topics()

len(model_min_20_ngram.vocab)

bigrams = []
for word in model_min_20_ngram.vocab:
    if len(word.split()) == 2:
        bigrams.append(word)

bigrams[:50]

documents, document_scores, document_ids = model_min_20_ngram.search_documents_by_topic(topic_num=1, num_docs=10)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()