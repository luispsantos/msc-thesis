import sys
sys.path.append("../")
import argparse
from tools.dictionary import Dictionary
from tools.utils import *
import torch
import os
from tools.lazy_reader import *
import math
import io
import gzip


parser = argparse.ArgumentParser()
parser.add_argument("--src_lang", type=str)
parser.add_argument("--tgt_lang", type=str)
parser.add_argument("--src_emb_path", type=str)
parser.add_argument("--tgt_emb_path", type=str)
parser.add_argument("--s2t_map_path", type=str, default=None)
parser.add_argument("--t2s_map_path", type=str, default=None)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--max_vocab", type=int, default=50000, help="number of most frequent embeddings to map")
parser.add_argument("--normalize_embeddings", type=str, default="double", choices=['',  'double', 'renorm', 'center', 'rescale'])
parser.add_argument("--smooth_c", type=float, default=0.01, help="smooth the empirical dist, the larger, the smoother")

args = parser.parse_args()

save_path = "../eval/"
if not os.path.exists(save_path):
    os.system("mkdir -p %s" % save_path)

assert args.src_emb_path is not None and args.tgt_emb_path is not None

args.cuda = torch.cuda.is_available()
device = torch.device('cuda') if args.cuda else torch.device("cpu")

src_dict, np_src_emb, np_src_freqs = load_embeddings(args, True)
tgt_dict, np_tgt_emb, np_tgt_freqs = load_embeddings(args, False)

gb_size = 1073741824
print("Size of the src and tgt embedding in Gigabytes: %f, %f" %
      ((np_src_emb.size * np_src_emb.itemsize / gb_size, np_tgt_emb.size * np_tgt_emb.itemsize / gb_size)))

# prepare embeddings
src_emb = torch.from_numpy(np_src_emb).float().to(device)
tgt_emb = torch.from_numpy(np_tgt_emb).float().to(device)

normalize_embeddings(src_emb, args.normalize_embeddings)
normalize_embeddings(tgt_emb, args.normalize_embeddings)

s2t = t2s = False
if args.s2t_map_path is not None:
    W_s2t = torch.from_numpy(torch.load(args.s2t_map_path)).to(device)
    s2t = True
if args.t2s_map_path is not None:
    W_t2s = torch.from_numpy(torch.load(args.t2s_map_path)).to(device)
    t2s = True
if not s2t and not t2s:
    exit(0)


def get_batches(n, batch_size):
    tot = math.ceil(n * 1.0 / batch_size)
    batches = []
    for i in range(tot):
        batches.append((i * batch_size, min((i + 1) * batch_size, n)))
    return batches


def map_embs(src_emb, tgt_emb, s2t=True, t2s=True):
    src2tgt_emb = tgt2src_emb = None

    if s2t:
        src_to_tgt_list = []
        for i, j in get_batches(src_emb.size(0), 512):
            src_emb_batch = src_emb[i:j, :]
            src_to_tgt = src_emb_batch.mm(W_s2t)
            src_to_tgt_list.append(src_to_tgt.cpu())
        src2tgt_emb = torch.cat(src_to_tgt_list, dim=0)
    if t2s:
        tgt_to_src_list = []
        for i, j in get_batches(tgt_emb.size(0), 512):
            tgt_emb_batch = tgt_emb[i:j, :]
            tgt_to_src = tgt_emb_batch.mm(W_t2s)
            tgt_to_src_list.append(tgt_to_src.cpu())
        tgt2src_emb = torch.cat(tgt_to_src_list, dim=0)

    return src2tgt_emb, tgt2src_emb


def identical_word_embedding(word, src_dict, tgt_dict, src_emb, tgt_emb):
    return tgt_emb


def export_embeddings(src_emb, tgt_emb, exp_path, src_dict, tgt_dict, s2t=True, t2s=True):
    mapped_src_emb, mapped_tgt_emb = map_embs(src_emb, tgt_emb, s2t=s2t, t2s=t2s)
    src_path = exp_path + f'{src_dict.lang}2{tgt_dict.lang}.fasttext.oov.vec.gz'
    tgt_path = exp_path + f'{tgt_dict.lang}2{src_dict.lang}.fasttext.oov.vec.gz'

    # word2idx dictionary for identical words in source and target languages
    src_same_words = {word: idx for word, idx in src_dict.word2id.items() if word in tgt_dict}
    tgt_same_words = {word: idx for word, idx in tgt_dict.word2id.items() if word in src_dict}

    # word2idx dictionary for non-identical words in source and target languages
    src_diff_words = {word: idx for word, idx in src_dict.word2id.items() if word not in tgt_dict}
    tgt_diff_words = {word: idx for word, idx in tgt_dict.word2id.items() if word not in src_dict}

    # same word order for identical source and target words (dicts are ordered in Python 3.6+)
    tgt_same_words = {word: tgt_same_words[word] for word in src_same_words}
    num_words = len(src_same_words) + len(src_diff_words) + len(tgt_diff_words)
    globals().update(locals())

    if s2t:
        mapped_src_emb = mapped_src_emb.numpy()
        print(f'Writing mapped source and target embeddings to {src_path}')
        with gzip.open(src_path, 'wb') as gzip_f:
            with io.TextIOWrapper(io.BufferedWriter(gzip_f, 8*1024*1024), encoding='utf-8') as embeddings_f:
                embeddings_f.write('{} {}\n'.format(num_words, src_emb.shape[1]))
                # write first the identical words in the source and target languages
                for (word, src_idx), (_, tgt_idx) in zip(src_same_words.items(), tgt_same_words.items()):
                    emb = identical_word_embedding(word, src_dict, tgt_dict,
                                                   mapped_src_emb[src_idx], tgt_emb[tgt_idx])
                    embeddings_f.write('{} {}\n'.format(word, ' '.join(f'{e:.4f}' for e in emb)))

                # write the unique words in the mapped source language
                for word, idx in src_diff_words.items():
                    embeddings_f.write('{} {}\n'.format(word, ' '.join(f'{e:.4f}' for e in mapped_src_emb[idx])))

                # write the unique words in the target language
                for word, idx in tgt_diff_words.items():
                    embeddings_f.write('{} {}\n'.format(word, ' '.join(f'{e:.4f}' for e in tgt_emb[idx])))

    if t2s:
        mapped_tgt_emb = mapped_tgt_emb.numpy()
        print(f'Writing mapped target to source embeddings to {tgt_path}')
        with io.open(tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % mapped_tgt_emb.shape)
            for i in range(len(tgt_dict)):
                f.write(u"%s %s\n" % (tgt_dict[i], " ".join('%.5f' % x for x in mapped_tgt_emb[i])))
        print(f'Writing corresponding source embeddings to {pre_src_path}')
        with io.open(pre_src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % src_emb.shape)
            for i in range(len(src_dict)):
                f.write(u"%s %s\n" % (src_dict[i], " ".join('%.5f' % x for x in src_emb[i])))

import time
start = time.time()
export_embeddings(src_emb, tgt_emb, save_path, src_dict, tgt_dict, s2t, t2s)
print('export took:', time.time() - start)
