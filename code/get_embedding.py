from gensim.models import Word2Vec
import pickle
import os
import numpy as np
import torch


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


def load_vocab_mapping(vocab_path):
    """
    Reads the vocab file and creates a dictionary mapping
    the integer index to the diagnosis string.
    """
    mapping = {}
    with open(vocab_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split("\t")
            index = int(tokens[0]) - 1
            diagnosis = tokens[1]
            mapping[index] = diagnosis
    return mapping


def load_vocab_dict(vocab_file):
    """
    Reads vocab_file => returns {word: index}, also pickles reverse mapping.
    """
    word_to_index = {}
    with open(vocab_file, "r") as f:
        for line in f:
            idx_str, token = line.strip().split("\t")
            word_to_index[token] = int(idx_str) - 1

    reverse_mapping = {v: k for k, v in word_to_index.items()}
    save_pkl("../resource/vocab.pkl", reverse_mapping)
    print(f"Vocab size: {len(word_to_index)}")
    return word_to_index


def build_corpus(pkl_paths, vocab_mapping):
    """
    Given a list of paths to pickled files (each containing a list of patients,
    where each patient is a list of visits, and each visit is a list of diagnosis indices),
    create a corpus where each "sentence" corresponds to a visit with diagnosis strings.
    """
    corpus = []
    looks = []
    for pkl_path in pkl_paths:
        if not os.path.exists(pkl_path):
            print(f"File not found: {pkl_path}")
            continue
        data = load_pkl(pkl_path)
        # Each element in data is a patient record (a list of visits)
        for patient in data:
            # Each visit is a list of diagnosis indices.
            for visit in patient:
                if visit:  # ensure the visit is not empty
                    # Map each index to its corresponding diagnosis string.
                    # If an index is not found in the mapping, use a placeholder "UNK".
                    sentence = []
                    for token in visit:
                        if token in vocab_mapping:
                            sentence.append(vocab_mapping[token])
                        else:
                            looks.append("{} found in visit: {}".format(token, visit))
                    corpus.append(sentence)
    return corpus, looks


def loadEmbeddingMatrix(wordvecFile, word_to_index, vocab_size):
    # Build reverse mapping: index -> word
    with open(wordvecFile, "r") as fw:
        # Read the header: total number of words and embedding dimension
        header = fw.readline().strip().split()
        total, dim = int(header[0]), int(header[1])
        # Initialize the embedding matrix with zeros
        W = np.zeros((vocab_size, dim), dtype=np.float32)
        for line in fw:
            parts = line.strip().split()
            # Reconstruct the word (in case it contains spaces)
            word = " ".join(parts[:-dim])
            vec = np.array(parts[-dim:], dtype=np.float32)
            try:
                token_value = word_to_index[
                    word
                ]  # Get the token index from the reverse mapping
            except KeyError:
                print(f"{word} is not in vocabulary; skipping.")
                continue
            # Adjust index if your vocab mapping is 1-indexed; here we subtract 1.
            W[token_value - 1] = vec
    return W


vocab_path = "../resource/vocab.txt"
wordvec_path = "../resource/word2vec.vector"
pkl_paths = [
    "../resource/X_train.pkl",
    "../resource/X_valid.pkl",
    "../resource/X_test.pkl",
]
# 1. Load the vocab mapping and build corpus to train
vocab_mapping = load_vocab_mapping(vocab_path)
vocab_size = len(vocab_mapping)
print("Loaded vocab mapping for {} words.".format(vocab_size))
corpus, looks = build_corpus(pkl_paths, vocab_mapping)
print("Number of sentences in corpus:", len(corpus))

# 2. Initialize and train the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
model.train(corpus, total_examples=len(corpus), epochs=10)

# 3. Save the word vectors in text format (word2vec.vector)
model.wv.save_word2vec_format(wordvec_path, binary=False)
