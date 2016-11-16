#Embedded file name: canonical_angles.py
import itertools
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

margin = 0.0001


def special_acos(sv):
    # numerical errors lead to values just above 1,
    # and arccos(1) = 0
    if sv >= 1:
        return 0
    else:
        return math.acos(sv)

def canonical_angles(embedding_a, embedding_b):
    """Computes the canonical angles between two embeddings.
    Inputs:
        embedding_a, an embedding A of shape (V, M)
        embedding_b, an embedding B of shape (V, N)
    Outputs:
        U, the transformation of embedding A to an orthonormal basis A'
        S, the singular values / canonical angles between bases A' and B'
        Vh, the transpose of the tranformation of B to an orthonormal basis B'
    """
    vocab_size_a, dim_size_a = embedding_a.shape
    vocab_size_b, dim_size_b = embedding_b.shape
    assert vocab_size_a == vocab_size_b, 'Vocab size mismatch'
    angle_matrix = np.dot(embedding_a.T, embedding_b)
    U, S, Vh = np.linalg.svd(angle_matrix)
    return (U, S, Vh)


def angle_spectrum(embedding_a, embedding_b):
    _, singular_values, _ = canonical_angles(embedding_a, embedding_b)
    angles = sorted([abs(math.acos(sv)) for sv in singular_values])
    return angles


def get_sum_squared_cosine_distance(embedding_a, embedding_b):
    _, singular_values, _ = canonical_angles(embedding_a, embedding_b)
    SSE = np.sum([ (1 - abs(s)) ** 2 for s in singular_values ])
    return SSE


def average_spectrum(embeddings):
    # For intra-treatment comparison
    angles_list = []
    for embedding_a, embedding_b in itertools.combinations(embeddings, 2):
        _, singular_values, _ = canonical_angles(embedding_a, embedding_b)
        angles_list.append(sorted([abs(special_acos(sv)) for sv in singular_values]))
    angle_averages = [np.average([l[i] for l in angles_list]) for i in range(len(angles_list[0]))]
    return angle_averages, angles_list


def get_max_principal_angle(embedding_a, embedding_b):
    _, singular_values, _ = canonical_angles(embedding_a, embedding_b)
    return max(singular_values)


def import_embedding(fname):
    index_to_embedding = []
    vocab_to_index = {}
    with open(fname) as f:
        first_line = f.readline()
        try:
            # word2vec format
            shape = [ int(i) for i in first_line.split() ]
            index_to_embedding = np.zeros(shape)
            for ind, line in enumerate(f):
                line_chunks = line.split()
                word = line_chunks[0]
                vals = [ float(i) for i in line_chunks[1:] ]
                index_to_embedding[ind, :] = vals
                vocab_to_index[word] = ind
        except:
            # GloVe format
            line_chunks = first_line.split()
            word = line_chunks[0]
            vals = [float(i) for i in line_chunks[1:]]
            index_to_embedding = [vals,]
            vocab_to_index[word] = 0
            for ind, line in enumerate(f):
                line_chunks = line.split()
                word = line_chunks[0]
                vals = [ float(i) for i in line_chunks[1:] ]
                index_to_embedding.append(vals)
                vocab_to_index[word] = ind + 1
            index_to_embedding = np.array(index_to_embedding)

    return (vocab_to_index, index_to_embedding)


def align_embeddings(vocabs_and_embeddings):
    """Aligns arbitrary numbers of embeddings by finding a common vocabulary
    and reordering each for that indexing"""
    vocab = set()
    for vocab_to_index, index_to_embedding in vocabs_and_embeddings:
        if len(vocab) == 0:
            vocab = set(vocab_to_index.iterkeys())
        else:
            vocab = vocab & set(vocab_to_index.iterkeys())

    new_vocab = list(vocab)
    new_embeddings = []
    for vocab_to_index, index_to_embedding in vocabs_and_embeddings:
        embedding_dim = index_to_embedding.shape[1]
        new_embedding = np.zeros((len(new_vocab), embedding_dim))
        for ind, word in enumerate(new_vocab):
            new_embedding[ind, :] = index_to_embedding[vocab_to_index[word], :]

        new_embeddings.append(new_embedding)

    return (new_vocab, new_embeddings)

def align_second_to_first(vocab_a, vocab_b, index_to_embedding):
    """Aligns a second embedding to have the same vocab-to-index as the first"""
    new_embedding = np.zeros((len(vocab_a), index_to_embedding.shape[1]))
    for word, ind in vocab_a.iteritems():
        new_embedding[ind, :] = index_to_embedding[vocab_b[word], :]
    return new_embedding


def main():
    plt.figure(figsize=(14,10))
    cmap = plt.get_cmap('gray')
    cNorm  = matplotlib.colors.Normalize(vmin=-2, vmax=55)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

    embedding_path_template = '/home/aks249/lola_embeddings/wiki_embeddings/wiki.txt-{}'
    in_vocab_to_index, in_embedding = import_embedding(embedding_path_template.format(0))
    q_in, _ = np.linalg.qr(in_embedding)
    for i in range(1, 50):
        print "Starting comparison {}...".format(i)
        out_vocab_to_index, out_embedding = import_embedding(embedding_path_template.format(i))
        out_embedding = align_second_to_first(in_vocab_to_index, out_vocab_to_index, out_embedding)
        q_out, _ = np.linalg.qr(out_embedding)
        angles_list = angle_spectrum(q_in, q_out)
        color = scalarMap.to_rgba(i)
        plt.plot(angles_list, color=color, label=str(i))
        q_in = q_out

    w2v_embedding_files = [
            '/share/magpie/projects/linear_word_embedding_compare/algs/outputs/rep-0-wiki.w2v',
            '/share/magpie/projects/linear_word_embedding_compare/algs/outputs/rep-1-wiki.w2v',
            '/share/magpie/projects/linear_word_embedding_compare/algs/outputs/rep-2-wiki.w2v'
        ]
    glove_embedding_files = [
            '/share/magpie/projects/linear_word_embedding_compare/algs/glove/GloVe-1.2/rep-0-wiki.glove-vectors.txt',
            '/share/magpie/projects/linear_word_embedding_compare/algs/glove/GloVe-1.2/rep-1-wiki.glove-vectors.txt',
            '/share/magpie/projects/linear_word_embedding_compare/algs/glove/GloVe-1.2/rep-2-wiki.glove-vectors.txt'
        ]
    # Add more embeddings here
    all_files = w2v_embedding_files + glove_embedding_files # Add more embedding lists here
    all_names = ['word2vec', 'GloVe']                       # Add more treatment names here
    name_colors = [':', '--']
    print "Loading embeddings..."
    unaligned_embeddings = []
    for embedding_file in all_files:
        unaligned_embeddings.append(import_embedding(embedding_file))
    print "Aligning embeddings..."
    new_vocab, embeddings = align_embeddings(unaligned_embeddings)
    print "Orthoganalizing..."
    qs = [np.linalg.qr(embedding)[0] for embedding in embeddings]
    print "Finding angles and plotting spectrum..."

    for i in range(len(all_names)):
        sim_angle_averages, sim_angle_list = average_spectrum(qs[i*3:(i+1)*3])
        plt.plot(
                sim_angle_averages,
                label=all_names[i],
                linestyle=name_colors[i],
                linewidth=2.0
        )

    plt.xlabel("Number of angles")
    plt.ylabel("Maximum angle")
    plt.savefig("iterating_spectrum.png")


if __name__ == '__main__':
    main()
