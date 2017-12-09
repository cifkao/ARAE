import argparse
import numpy as np
import random
import sys

import torch
from torch.autograd import Variable

from models import load_models, generate
from utils import to_gpu, Corpus, batchify


def main(args):
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    ###########################################################################
    # Load the models
    ###########################################################################

    model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.load_path)

    ###########################################################################
    # Load the data
    ###########################################################################

    corpus = Corpus(args.data_path,
                    maxlen=100,
                    vocab_size=model_args['vocab_size'],
                    lowercase=model_args['lowercase'])
    eval_batch_size = 10
    test_data = batchify(corpus.test, eval_batch_size, shuffle=False)


    ###########################################################################
    # Prediction code
    ###########################################################################

    autoencoder.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    num_examples = 0
    criterion_ce = torch.nn.CrossEntropyLoss(size_average=False)
    for i, batch in enumerate(test_data):
        source, target, lengths = batch
        source = Variable(source, volatile=True)
        target = Variable(target, volatile=True)
        #source = to_gpu(args.cuda, Variable(source, volatile=True))
        #target = to_gpu(args.cuda, Variable(target, volatile=True))

        # Generate output.

        # output: batch x seq_len x ntokens
        hidden = autoencoder(source, lengths, noise=False, encode_only=True)
        max_indices = autoencoder.generate(hidden, model_args['maxlen'], sample=False)

        # Decode in training mode to compute loss.

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=False)
        flattened_output = output.view(-1, ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += criterion_ce(masked_output/model_args['temp'], masked_target).data
        num_examples += source.size()[0]

        aeoutf = args.outf
        with open(aeoutf, "a") as f:
            max_indices = max_indices.view(eval_batch_size, -1).data.cpu().numpy()
            target = target.view(eval_batch_size, -1).data.cpu().numpy()
            eos = corpus.dictionary.word2idx['<eos>']
            for t, idx in zip(target, max_indices):
                # real sentence
                length = list(t).index(eos) if eos in t else len(t)
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t[:length]])
                f.write(chars)
                f.write("\t")
                # autoencoder output sentence
                length = list(idx).index(eos) if eos in idx else len(idx)
                chars = " ".join([corpus.dictionary.idx2word[x] for x in idx[:length]])
                f.write(chars)
                f.write("\n")

    print("Processed {} examples".format(num_examples))
    print("Cross-entropy: {:.4f}".format((total_loss / num_examples)[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    parser.add_argument('--load_path', type=str, required=True,
                        help='directory to load models from')
    parser.add_argument('--data_path', type=str, required=True,
                        help='directory to load data from')
    parser.add_argument('--temp', type=float, default=1,
                        help='softmax temperature (lower --> more discrete)')
    parser.add_argument('--outf', type=str, default='./autoencoder.txt',
                        help='filename and path to write to')
    parser.add_argument('--noprint', action='store_true',
                        help='prevents examples from printing')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nocuda', action='store_false', dest='cuda',
                        help='do not use CUDA')
    args = parser.parse_args()
    print(vars(args))
    main(args)
