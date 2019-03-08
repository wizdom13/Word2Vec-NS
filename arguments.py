import argparse
import torch

def preprocess_args():
    parser = argparse.ArgumentParser(description='Word2Vec preprocess')
    parser.add_argument('--data_dir', type=str, default='./data/', 
                        help="Data directory path (default: ./data/)")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', 
                        help="Corpus path for building vocab (default: ./data/corpus.txt)")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', 
                        help="Corpus path (default: ./data/corpus.txt)")
    parser.add_argument('--unk', type=str, default='<UNK>', 
                        help="UNK token (default: <UNK>) ")
    parser.add_argument('--window', type=int, default=5, 
                        help="Window size (default: 5)")
    parser.add_argument('--max-vocab', type=int, default=20000, 
                        help="Maximum number of vocab (default: 20000)")

    args = parser.parse_args()  
    return args


def train_args():
    parser = argparse.ArgumentParser(description='Word2Vec training')
    parser.add_argument('--name', type=str, default='SkipGramNS',
                        help="Model name")
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help="Data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/',
                        help="Model directory path")
    parser.add_argument('--log-dir', default=None,
                        help='Directory to save agent logs (default: runs/CURRENT_DATETIME_HOSTNAME)')
    parser.add_argument('--e_dim', type=int, default=300,
                        help="Embedding dimension")
    parser.add_argument('--n_negs', type=int, default=20,
                        help="Number of negative samples")
    parser.add_argument('--epoch', type=int, default=100,
                        help="Number of epochs")
    parser.add_argument('--mb', type=int, default=4096,
                        help="Mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5,
                        help="Subsample threshold")
    parser.add_argument('--resume', action='store_true',
                        help="Continue learning")
    parser.add_argument('--weights', action='store_true',
                        help="Use weights for negative sampling")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Use GPU training (default: True)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("GPU training: ", args.cuda)
    return args    


def plot_args():
    parser = argparse.ArgumentParser(description='Word2Vec training')
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--result_dir', type=str, default='./result/', help="result directory path")
    parser.add_argument('--model', type=str, default='tsne', choices=['pca', 'tsne'], help="model for visualization")
    parser.add_argument('--top_k', type=int, default=1000, help="scatter top-k words")
    return parser.parse_args()