import argparse
from typing import Optional
import torch
import csv
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from load_data import load_data_word2vec
from utils import get_similarity_scores, compute_spearman_correlation


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Word2vec(nn.Module):
    """
    Word2vec model (here, skip-gram with negative sampling)
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initializes the Word2vec model.
        You may add optional keyword arguments to this function if needed.

        Inputs:
            vocab_size (int): Size of the vocabulary
            embed_dim (int): Dimensionality of the word embeddings
        """
        # TODO
        super(Word2vec, self).__init__()
        self.input_embedding = nn.Embedding(vocab_size+1, embed_dim)
        self.context_embedding = nn.Embedding(vocab_size+1, embed_dim)

    def forward(self, input_tokens: torch.Tensor, context_tokens: torch.Tensor, negative_context: Optional[torch.Tensor] = None):
        """
        Forward pass of the model to compute the embeddings.

        Inputs:
            input_tokens (torch.Tensor): Input words (w)
            context_tokens (torch.Tensor): Context words (c)
            negative_context (torch.Tensor): Negative context word

        Outputs:
            input_tokens_embeds (torch.Tensor): Embeddings of input words
            context_embeds (torch.Tensor): Embeddings of context words
            negative_embeds (torch.Tensor): Embeddings of negative context words
        """
        # TODO
        input_tokens_embeds = self.input_embedding(input_tokens)
        context_embeds = self.context_embedding(context_tokens)
        if negative_context is not None:
            negative_embeds = self.context_embedding(negative_context)
            return input_tokens_embeds, context_embeds, negative_embeds
        return input_tokens_embeds, context_embeds

    def compute_loss(self, input_embeds: torch.Tensor, context_embeds: torch.Tensor, negative_embeds: Optional[torch.Tensor] = None):
        """
        Computes the loss using the embeddings from the forward pass. If negative_embeds is not None,
        it includes the loss from negative sampling.

        Inputs:
            input_embeds (torch.Tensor): Embeddings of input words (w)
            context_embeds (torch.Tensor): Embeddings of context words (c)
            negative_embeds (torch.Tensor): Embeddings of negative context words

        Outputs:
            loss (torch.Tensor)
        """
        # TODO
        p = torch.matmul(input_embeds, context_embeds.T)
        pos_loss = F.log_softmax(p, dim=1).sum(dim=1).mean()
        if negative_embeds is not None:
            n = torch.matmul(input_embeds, negative_embeds.T)
            neg_loss = F.log_softmax(-n, dim=1).sum(dim=1).mean()
        return -(pos_loss + neg_loss) if negative_embeds is not None else -pos_loss

    def pred(self, input_tokens: torch.Tensor):
        """
        Predicts the embeddings of the input tokens.

        Inputs:
            input_tokens (torch.Tensor): Input words

        Outputs:
            embeds (torch.Tensor): Embeddings of input words
        """
        # TODO
        return self.input_embedding(input_tokens)

    def learn(self, train_data, dataset, num_epochs: int, loss_func, optimizer, vocab_size, num_neg_samples):
        """
        Training word2vec model.
        """
        # TODO
        for epoch in range(num_epochs):
            total_loss = 0.0
            count = 1
            for batch_idx, (target, context) in enumerate(train_data):
                optimizer.zero_grad()
                samples = list(set(range(vocab_size)) - set(target.numpy().tolist()) - set(context.numpy().tolist()) - set([-1]))
                negative = torch.tensor(random.sample(samples, num_neg_samples))
                input_embeddings, context_embeddings, negative_embeddings = self.forward(target, context, negative)
                loss = self.compute_loss(input_embeddings, context_embeddings, negative_embeddings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (batch_idx + 1) % 32 == 0:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx//32}, % Done {100 * count/len(train_data)}, Loss: {total_loss / 100:.4f}")
                    total_loss = 0.0
                count += 1


def get_args():
    parser = argparse.ArgumentParser(description='word2vec model')
    parser.add_argument('-a', '--additional_data', action='store_true',
                        help='Include additional data for training')

    parser.add_argument('-d', '--embed_dim', type=int, default=300,
                        help='Dimensionality of the word embeddings')
    parser.add_argument('-w', '--window_size', type=int, default=5,
                        help='Size of the context window')
    parser.add_argument('-wt', '--window_type', type=str, default='linear',
                        help='Type of the context window')
    parser.add_argument('-n', '--num_neg_samples', type=int, default=5,
                        help='(For negative sampling) number of negative samples to use')
    parser.add_argument('-ne', '--neg_exponent', type=float, default=0.75,
                        help='(For negative sampling) exponent for negative sampling distribution')
    parser.add_argument('--min-freq', type=int, default=5,
                        help='The minimum frequency of words to consider')

    parser.add_argument('-ep', '--num_epochs', type=int, default=1,
                        help='Number of training epochs')

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    additional_data = args.additional_data
    embed_dim = args.embed_dim
    window_size = args.window_size
    window_type = args.window_type
    num_neg_samples = args.num_neg_samples
    neg_exponent = args.neg_exponent
    min_freq = args.min_freq
    num_epochs = args.num_epochs
    experiment_name = args.experiment_name

    # Load data
    train_dataset, train_loader, cont_dev_data, cont_test_data, isol_dev_data, isol_test_data = load_data_word2vec(window_size, num_neg_samples, min_freq, neg_exponent, additional_data)

    vocab_size = train_dataset.vocab_size
    print("Vocab Size: ", vocab_size)

    model = Word2vec(vocab_size, embed_dim)
    model = model.to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    model.learn(train_loader, train_dataset, num_epochs, loss_func, optimizer, vocab_size, num_neg_samples)

    # Compute the embeddings for dev/test words (using the adapted inputs)
    isol_dev_embeds_word1 = model.pred(isol_dev_data[0])
    isol_dev_embeds_word2 = model.pred(isol_dev_data[1])
    isol_test_embeds_word1 = model.pred(isol_test_data[0])
    isol_test_embeds_word2 = model.pred(isol_test_data[1])
    cont_dev_embeds_word1 = model.pred(cont_dev_data[0])
    cont_dev_embeds_word2 = model.pred(cont_dev_data[1])
    cont_test_embeds_word1 = model.pred(cont_test_data[0])
    cont_test_embeds_word2 = model.pred(cont_test_data[1])

    # Save the embeddings to text file
    def save_embeddings(file_path, index_to_word, data, word_embeddings):
        with open(file_path, "w") as file:
            for i, idx in enumerate(data):
                word = index_to_word[idx.item()]
                embedding = ' '.join(map(str, word_embeddings[i].tolist()))
                file.write(f"{word} {embedding}\n")

    save_embeddings("./results/word2vec_cont_test_words1_embeddings.txt", train_dataset.value_to_word, cont_test_data[0], cont_test_embeds_word1)
    save_embeddings("./results/word2vec_cont_test_words2_embeddings.txt", train_dataset.value_to_word, cont_test_data[1], cont_test_embeds_word2)
    save_embeddings("./results/word2vec_isol_test_words1_embeddings.txt", train_dataset.value_to_word, isol_test_data[0], isol_test_embeds_word1)
    save_embeddings("./results/word2vec_isol_test_words2_embeddings.txt", train_dataset.value_to_word, isol_test_data[1], isol_test_embeds_word2)

    # Compute word pair similarity scores using the embeddings
    isol_dev_sim_scores = get_similarity_scores(isol_dev_embeds_word1.detach().numpy(), isol_dev_embeds_word2.detach().numpy())
    cont_dev_sim_scores = get_similarity_scores(cont_dev_embeds_word1.detach().numpy(), cont_dev_embeds_word2.detach().numpy())

    # Evaluate the similarity scores against human ratings
    # Read the labels
    def load_y_csv_file(file_path: str):
        labels = []
        print(f"Loading CSV file from {file_path}...")
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'id':
                    continue
                labels.append(float(row[1]))
        return labels

    isol_dev_labels = load_y_csv_file("data/isolated_similarity/isolated_dev_y.csv")
    cont_dev_labels = load_y_csv_file("data/contextual_similarity/contextual_dev_y.csv")

    # Compute the scores
    isol_dev_corr = compute_spearman_correlation(isol_dev_sim_scores, isol_dev_labels)
    cont_dev_corr = compute_spearman_correlation(cont_dev_sim_scores, cont_dev_labels)

    print("Evaluating on: word pairs in isolation")
    print("Correlation score on dev set:", isol_dev_corr)

    print("\nEvaluating on: word pairs in context")
    print("Correlation score on dev set:", cont_dev_corr)


if __name__ == "__main__":
    main()
