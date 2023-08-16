# DBN
This repository contains both tensorflow and pytorch implementations of TKDE-2023 paper "[Topic Modeling on Document Networks with Dirichlet Optimal Transport Barycenter](/paper/TKDE23-DBN.pdf)", authored by [Delvin Ce Zhang](http://delvincezhang.com/) and [Hady W. Lauw](http://www.hadylauw.com/home).

DBN is a Graph Neural Network model designed for interconnected texts in a graph structure, such as academic citation graphs and Webpage hyperlink graphs. DBN designs an Optimal Transport Barycenter to capture graph connectivity, and incorporates pre-trained word embeddings to alleviate short text sparsity problem. The topic modeling decoder makes the learned text embeddings semantically interpretable.

## Implementation Environment
- python == 3.10
- numpy == 1.20.3
- pytorch == 2.0.0 (for pytorch version)
- tensorflow == 1.15.0 (for tensorflow version)

## Run
Please note that the results reported in the paper are produced by tensorflow version. We reimplement the model using pytorch upon publication, which can reproduce most of the results. Different libraries may result in slight deviations of hyperparameters.

Below command lines are runnable for both tensorflow and pytorch implementations.

`python main.py -mn dbn`  # run DBN

`python main.py -mn d2bn`   # run D2BN

### Parameter Setting
Below hyperparameters are shared by both tensorflow and pytorch implementations.
- -mn: model name, default = dbn (set `dbn` for DBN, and `d2bn` for D2BN)
- -ne: number of training epochs, default = 100 (set 100 for small datasets (ds, ml, pl), and 10 for large datasets (aminer and web))
- -no: number of optimal transport iterations, default = 10
- -lr: learning rate, default = 0.002
- -ms: minibatch size, default = 64
- -dn: dataset name, default = ml
- -nt: number of topics (dimension of text embeddings), default = 64 (K in the paper)
- -reg_ot: regularizer for optimal transport, default = 2 ($\lambda_{OT}$ in the paper)
- -reg_str: regularizer for structural decoding, default = 1 ($\lambda_s$ in the paper. This -reg_str is useful only if -mn is set to `d2bn`. If -mn is set to `dbn`, this -reg_str can be ignored.)
- -reg_kld: regularizer for KL divergence, default = 0.001 ($\lambda_{KL}$ in the paper)
- -dp: concentration parameter of dirichlet prior, default = 1 ($\alpha^0$ in the paper)
- -tr: training ratio, the ratio of training documents to the total documents, default = 0.8
- -nl: number of graph convolutional layers, default = 3
- -nn: number of sampled neighbors for aggregation, default = 5
- -rs: random seed
- -gpu: gpu

## Data
We release DS, ML, PL, and Aminer datasets in `./data` folder. For all these four datasets, please unzip the `.zip` file and put the unzipped file into `./data` folder. For the largest Web dataset, please email Delvin Ce Zhang (delvincezhang@gmail.com) for access.

Each dataset contains contents, links, vocabulary, pretrained glove word embeddings, pretrained deepwalk structure embeddings, labels, and label names.

- contents: each row corresponds to a document, containing a sequence of words represented by word IDs in vocabulary. Word ID starts from 0. For example, a row with `[0, 6, 4]` means a document with a sequence of three words, i.e., the 0th, 6th, and 4th words in the vocabulary. There are N rows (N documents) in total.
- links: each row corresponds to a link represented by a pair of document IDs. For example, a row  with `[5, 8]` means a link from document 5 to document 8.
- voc (|V|x1): vocabulary.
- pretrained glove word embeddings (|V|x300): 300-dimensional pretrained glove word embeddings, with each row corresponding to a word in the vocabulary. For example, the 0th word embedding corresponds to the 0th word in the vocabulary.
- pretrained deepwalk structure embeddings (Nx128): 128-dimensional pretrained deepwalk structure embeddings, with each row corresponding to a document. For example, the 0th structure embedding corresponds to the 0th document.
- labels (Nx1): labels or categories of N documents. For example, the 0th label corresponds to the 0th document.
- label names: the names of labels or categories.

## Output
We output two files for each training. Output results are saved to the `./dataset_name/results` folder.

- doc_topic_dist.txt (NxK): each row is a K-dimensional document embedding, representing document-topic distribution. There are N documents in total. The first 80% embeddings are for training documents, and the remaining 20% embeddings are for testing documents. These document embeddings can fulfill downstream tasks, including text classification and graph link prediction.
- topic_word_dist.txt (Kx|V|): topic-word distribution, where each row is a probability distribution over the whole vocabulary. There are K rows (K topics). This topic-word distribution can fulfill topic analysis tasks, including topic coherence.

## Reference
If you find our paper useful, including code and data, please cite

```
@article{dbn,
  title={Topic Modeling on Document Networks with Dirichlet Optimal Transport Barycenter},
  author={Zhang, Delvin Ce and Lauw, Hady W},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```
