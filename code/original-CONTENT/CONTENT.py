#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example. Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.

Updated to use PyTorch instead of the deprecated Theano/Lasagne.
'''

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import Config
from patient_data_reader import PatientReader
import os
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score as pr_auc

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 200

# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
num_epochs = 6


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    if shuffle:
        indices = np.arange(len(inputs[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


def loadEmbeddingMatrix(wordvecFile):
    fw = open(wordvecFile, "r")
    headline = fw.readline().strip().split()
    vocabSize = int(headline[0])
    dim = int(headline[1])
    W = np.zeros((vocabSize, dim)).astype(np.float32)
    for line in fw:
        tabs = line.strip().split()
        vec = np.asarray([float(x) for x in tabs[1:]], dtype=np.float32)
        ind = int(tabs[0])
        W[ind - 1] = vec
    fw.close()
    return W


# Define custom PyTorch modules to replace Lasagne's layers
class MaskingLayer(nn.Module):
    def __init__(self):
        super(MaskingLayer, self).__init__()

    def forward(self, x, mask):
        # Apply mask to the input tensor
        return x * mask.unsqueeze(-1)


class ThetaLayer(nn.Module):
    def __init__(self, maxlen):
        super(ThetaLayer, self).__init__()
        self.maxlen = maxlen
        self.klterm = 0
        self.theta = None

    def forward(self, mu, log_sigma):
        batch_size, n_topics = mu.size()

        # Reparameterization trick
        epsilon = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * log_sigma) * epsilon

        # Apply softmax to get theta
        theta = F.softmax(z, dim=1)

        # Save theta for later use
        self.theta = theta

        # Compute KL divergence
        self.klterm = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        # Expand theta to match maxlen
        expanded_theta = theta.unsqueeze(1).expand(-1, self.maxlen, -1)

        return expanded_theta


class ContentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_topics, max_length):
        super(ContentModel, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_topics = n_topics
        self.max_length = max_length

        # Embedding layer
        self.embed = nn.Linear(vocab_size, embed_size, bias=False)

        # GRU layer
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

        # Topic model layers
        self.dense1 = nn.Linear(vocab_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, n_topics)
        self.log_sigma = nn.Linear(hidden_size, n_topics)

        # Output layers
        self.B = nn.Linear(vocab_size, n_topics, bias=False)
        self.output_dense = nn.Linear(hidden_size, 1)

        # Theta layer
        self.theta_layer = ThetaLayer(max_length)

    def forward(self, x, mask):
        batch_size, seq_len, vocab_size = x.size()

        # Embedding
        embedded = self.embed(x)

        # Apply mask to embedded
        embedded = embedded * mask.unsqueeze(-1)

        # GRU layer
        gru_out, h_n = self.gru(embedded)

        # Topic model
        h1 = F.relu(self.dense1(x))
        h2 = F.relu(self.dense2(h1))
        mu = self.mu(h2.mean(dim=1))  # Average over sequence length
        log_sigma = self.log_sigma(h2.mean(dim=1))

        # Get theta
        theta_expanded = self.theta_layer(mu, log_sigma)

        # Context vector
        B_out = self.B(x)  # [batch, seq_len, n_topics]
        context = (B_out * theta_expanded).mean(dim=-1)  # [batch, seq_len]

        # Output predictions
        dense_out = self.output_dense(gru_out).squeeze(-1)  # [batch, seq_len]
        out = dense_out + context
        out = torch.sigmoid(out)

        # Apply mask to output
        out = out * mask + 1e-6

        return out, h_n, self.theta_layer.theta


def main(data_sets, W_embed):
    # Learning rate
    learning_rate = 0.001

    # Min/max sequence length
    MAX_LENGTH = 300

    # Get training data
    X_raw_data, Y_raw_data = data_sets.get_data_from_type("train")
    trainingAdmiSeqs, trainingMask, trainingLabels, trainingLengths, ltr = prepare_data(X_raw_data, Y_raw_data,
                                                                                        vocabsize=491,
                                                                                        maxlen=MAX_LENGTH)
    Num_Samples, MAX_LENGTH, N_VOCAB = trainingAdmiSeqs.shape

    # Get validation data
    X_valid_data, Y_valid_data = data_sets.get_data_from_type("valid")
    validAdmiSeqs, validMask, validLabels, validLengths, lval = prepare_data(X_valid_data, Y_valid_data, vocabsize=491,
                                                                             maxlen=MAX_LENGTH)

    # Get test data
    X_test_data, Y_test_data = data_sets.get_data_from_type("test")
    test_admiSeqs, test_mask, test_labels, testLengths, ltes = prepare_data(X_test_data, Y_test_data, vocabsize=491,
                                                                            maxlen=MAX_LENGTH)

    # Print statistics
    alllength = sum(trainingLengths) + sum(validLengths) + sum(testLengths)
    print(f"Total sequence length: {alllength}")
    eventNum = sum(ltr) + sum(lval) + sum(ltes)
    print(f"Total event count: {eventNum}")

    print("Building network ...")
    N_BATCH = 1
    embedsize = 100
    n_topics = 50

    # Create model
    model = ContentModel(
        vocab_size=N_VOCAB,
        embed_size=embedsize,
        hidden_size=N_HIDDEN,
        n_topics=n_topics,
        max_length=MAX_LENGTH
    ).to(device)

    # If we have pretrained embeddings, use them
    if W_embed is not None:
        with torch.no_grad():
            model.embed.weight.copy_(torch.from_numpy(W_embed.T))

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create output directory
    os.makedirs("theta_with_rnnvec", exist_ok=True)

    print("Training ...")
    try:
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            thetas_train = []

            # Set model to training mode
            model.train()

            for batch in iterate_minibatches_listinputs([trainingAdmiSeqs, trainingLabels, trainingMask], N_BATCH,
                                                        shuffle=True):
                inputs = batch

                # Convert numpy arrays to torch tensors
                x = torch.FloatTensor(inputs[0]).to(device)
                y = torch.FloatTensor(inputs[1]).to(device)
                mask = torch.FloatTensor(inputs[2]).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs, h_n, theta = model(x, mask)

                # Compute loss
                loss = F.binary_cross_entropy(outputs.flatten(), y.flatten(), reduction='sum')
                loss += model.theta_layer.klterm

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                # Save statistics
                train_err += loss.item()
                train_batches += 1

                # Save theta and RNN vectors
                rnnvec_train = h_n.squeeze(0).detach().cpu().numpy()
                theta_train = theta.detach().cpu().numpy()
                rnnout_train = np.concatenate([theta_train, rnnvec_train], axis=1)
                thetas_train.append(rnnout_train.flatten())

                if (train_batches + 1) % 1000 == 0:
                    print(train_batches)

            # Save thetas
            np.save(f"theta_with_rnnvec/thetas_train{epoch}", thetas_train)

            # Print epoch statistics
            print(f"Epoch {epoch + 1} of {num_epochs} took {time.time() - start_time:.3f}s")
            print(f"  training loss:\t\t{train_err / train_batches:.6f}")

            # Evaluate on test set
            test_err = 0
            test_batches = 0
            new_testlabels = []
            pred_testlabels = []
            thetas = []

            # Set model to evaluation mode
            model.eval()

            with torch.no_grad():
                for batch in iterate_minibatches_listinputs([test_admiSeqs, test_labels, test_mask, testLengths], 1,
                                                            shuffle=False):
                    inputs = batch

                    # Convert numpy arrays to torch tensors
                    x = torch.FloatTensor(inputs[0]).to(device)
                    y = torch.FloatTensor(inputs[1]).to(device)
                    mask = torch.FloatTensor(inputs[2]).to(device)
                    leng = inputs[3][0]

                    # Forward pass
                    outputs, h_n, theta = model(x, mask)

                    # Compute loss
                    loss = F.binary_cross_entropy(outputs.flatten(), y.flatten(), reduction='sum')
                    loss += model.theta_layer.klterm

                    # Save statistics
                    test_err += loss.item()
                    test_batches += 1

                    # Save predictions and labels
                    new_testlabels.extend(y.cpu().numpy().flatten()[:leng])
                    pred_testlabels.extend(outputs.cpu().numpy().flatten()[:leng])

                    # Save theta and RNN vectors
                    rnnvec = h_n.squeeze(0).cpu().numpy()
                    theta_np = theta.cpu().numpy()
                    rnnout = np.concatenate([theta_np, rnnvec], axis=1)
                    thetas.append(rnnout.flatten())

            # Compute metrics
            test_auc = roc_auc_score(new_testlabels, pred_testlabels)
            test_pr_auc = pr_auc(new_testlabels, pred_testlabels)
            test_pre_rec_f1 = precision_recall_fscore_support(
                np.array(new_testlabels),
                np.array(pred_testlabels) > 0.5,
                average='binary'
            )
            test_acc = accuracy_score(np.array(new_testlabels), np.array(pred_testlabels) > 0.5)

            # Print results
            print("Final results:")
            print(f"  test loss:\t\t{test_err / test_batches:.6f}")
            print(f"  test auc:\t\t{test_auc:.6f}")
            print(f"  test pr_auc:\t\t{test_pr_auc:.6f}")
            print(f"  test accuracy:\t\t{test_acc * 100:.2f} %")
            print(
                f"  test Precision, Recall and F1:\t\t{test_pre_rec_f1[0]:.4f} %\t\t{test_pre_rec_f1[1]:.4f}\t\t{test_pre_rec_f1[2]:.4f}")

    except KeyboardInterrupt:
        pass


def prepare_data(seqs, labels, vocabsize, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    eventSeq = []

    for seq in seqs:
        t = []
        for visit in seq:
            t.extend(visit)
        eventSeq.append(t)
    eventLengths = [len(s) for s in eventSeq]

    if maxlen is not None:
        new_seqs = []
        new_lengths = []
        new_labels = []
        for l, s, la in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_lengths.append(l)
                new_labels.append(la)
            else:
                new_seqs.append(s[:maxlen])
                new_lengths.append(maxlen)
                new_labels.append(la[:maxlen])
        lengths = new_lengths
        seqs = new_seqs
        labels = new_labels

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen, vocabsize)).astype('float32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    y = np.ones((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(seqs):
        x_mask[idx, :lengths[idx]] = 1
        for j, sj in enumerate(s):
            for tsj in sj:
                x[idx, j, tsj - 1] = 1
    for idx, t in enumerate(labels):
        y[idx, :lengths[idx]] = t

    return x, x_mask, y, lengths, eventLengths


def eval(epoch):
    new_testlabels = np.load(f"CONTENT_results/testlabels_{epoch}.npy")
    pred_testlabels = np.load(f"CONTENT_results/predlabels_{epoch}.npy")
    test_auc = roc_auc_score(new_testlabels, pred_testlabels)
    test_pr_auc = pr_auc(new_testlabels, pred_testlabels)
    test_acc = accuracy_score(new_testlabels, pred_testlabels > 0.5)
    print(f'AUC: {test_auc:.4f}')
    print(f'PRAUC: {test_pr_auc:.4f}')
    print(f'ACC: {test_acc:.4f}')

    pre, rec, threshold = precision_recall_curve(new_testlabels, pred_testlabels)
    test_pre_rec_f1 = precision_recall_fscore_support(new_testlabels, pred_testlabels > 0.5, average='binary')
    print(
        f"  test Precision, Recall and F1:\t\t{test_pre_rec_f1[0]:.4f} %\t\t{test_pre_rec_f1[1]:.4f}\t\t{test_pre_rec_f1[2]:.4f}")

    epoch = 6
    rnn_testlabels = np.load(f"rnn_results/testlabels_{epoch}.npy")
    rnn_pred_testlabels = np.load(f"rnn_results/predlabels_{epoch}.npy")
    pre_rnn, rec_rnn, threshold_rnn = precision_recall_curve(rnn_testlabels, rnn_pred_testlabels)
    test_pre_rec_f1 = precision_recall_fscore_support(rnn_testlabels, rnn_pred_testlabels > 0.5, average='binary')
    test_auc = roc_auc_score(rnn_testlabels, rnn_pred_testlabels)
    test_acc = accuracy_score(rnn_testlabels, rnn_pred_testlabels > 0.5)
    print(f'rnnAUC: {test_auc:.4f}')
    print(f'rnnACC: {test_acc:.4f}')
    print(
        f"  rnn test Precision, Recall and F1:\t\t{test_pre_rec_f1[0]:.4f} %\t\t{test_pre_rec_f1[1]:.4f}\t\t{test_pre_rec_f1[2]:.4f}")

    epoch = 5
    wv_testlabels = np.load(f"rnnwordvec_results/testlabels_{epoch}.npy")
    wv_pred_testlabels = np.load(f"rnnwordvec_results/predlabels_{epoch}.npy")
    pre_wv, rec_wv, threshold_wv = precision_recall_curve(wv_testlabels, wv_pred_testlabels)
    test_pre_rec_f1 = precision_recall_fscore_support(new_testlabels, wv_pred_testlabels > 0.5, average='binary')
    test_auc = roc_auc_score(wv_testlabels, wv_pred_testlabels)
    test_acc = accuracy_score(wv_testlabels, wv_pred_testlabels > 0.5)
    print(f'wvAUC: {test_auc:.4f}')
    print(f'wvACC: {test_acc:.4f}')
    print(
        f"  wv test Precision, Recall and F1:\t\t{test_pre_rec_f1[0]:.4f} %\t\t{test_pre_rec_f1[1]:.4f}\t\t{test_pre_rec_f1[2]:.4f}")

    import matplotlib.pyplot as plt
    plt.plot(rec, pre, label='CONTENT')
    plt.plot(rec_rnn, pre_rnn, label='RNN')
    plt.plot(rec_wv, pre_wv, label='RNN+word2vec')
    plt.legend()
    plt.title("Precision-Recall Curves")
    plt.show()


def list2dic(_list):
    output = dict()
    for i in _list:
        if i in output:
            output[i] += 1
        else:
            output[i] = 0
    return output


def outputCodes(indexs, patientList):
    HightPat = []
    for i in indexs:
        HightPat.extend(patientList[i])
    high = list2dic(HightPat)
    items = sorted(high.items(), key=lambda d: d[1], reverse=True)
    for key, value in items[:20]:
        print(key, value)


def scatter(x, colors):
    import matplotlib.patheffects as PathEffects
    import seaborn as sns
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 50))
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    # We add the labels for each digit.
    txts = []
    return f, ax, sc, txts


def clustering(thetaPath, dataset):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.manifold import TSNE

    thetas = np.asarray(np.load(thetaPath))[:, 50:]
    ypred = MiniBatchKMeans(n_clusters=20).fit_predict(thetas).flatten()
    tsn = TSNE(random_state=256, n_iter=2000).fit_transform(thetas)
    scatter(tsn, ypred)
    plt.show()

    X_test_data, Y_test_data = dataset.get_data_from_type("test")
    new_X = []
    for s in X_test_data:
        ss = []
        for t in s:
            ss.extend(t)
        new_X.append(ss)

    print("\n")
    for ylabel in range(20):
        indexs = np.where(ypred == ylabel)[0]
        print("Cluster", ylabel)
        outputCodes(indexs, new_X)
        n = []
        for i in indexs:
            n.append(sum(Y_test_data[i]))
        n = np.array(n)
        aveCount = np.mean(n)
        stdev = np.std(n)
        print("Number of Examples:\t", len(indexs))
        print("Readmission AveCount:\t", aveCount)
        print("Readmission Std:\t", stdev)
        print("\n")


if __name__ == '__main__':
    FLAGS = Config()
    data_sets = PatientReader(FLAGS)
    wordvecPath = os.path.join(FLAGS.data_path, "word2vec.vector")

    if os.path.exists(wordvecPath):
        W_embed = loadEmbeddingMatrix(wordvecPath)
    else:
        print(f"Warning: Word embedding file not found at {wordvecPath}")
        W_embed = None

    # Choose which function to run
    # Uncomment the function you want to use
    #main(data_sets, W_embed)
    # eval(2)
    # thetaPath = "theta_with_rnnvec/thetas_train0.npy"
    # clustering(thetaPath, data_sets)
