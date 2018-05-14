# BiLSTM with BF-max pooling for Universal sentence representations

*BiLSTM with BF-max pooling* is a *sentence embeddings* method that provides semantic sentence representations. It is trained on natural language inference data and generalizes well to many different tasks.
*BiLSTM with BF-max pooling* is inspired by [InferSent](https://arxiv.org/abs/1705.02364),we use an imporved pooling mechanisim to boost the BiLSTM with max pooling. 



## Dependencies

This code is written in python. The dependencies are:

* Python 2.7 (with recent versions of [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) >= 0.12
* NLTK >= 3

## Download datasets
To get GloVe, SNLI and MultiNLI [2GB, 90MB, 216MB], run (in dataset/):
```bash
./get_data.bash
```
This will download GloVe and preprocess SNLI/MultiNLI datasets. For MacOS, you may have to use *p7zip* instead of *unzip*.
## Train model on Natural Language Inference (SNLI)
To reproduce our results and train our models on [SNLI](https://nlp.stanford.edu/projects/snli/), set **GLOVE_PATH** in *train_nli.py*, then run:
```bash
python train_nli.py --bf=1 use bf-balanced(1) or --bf=2 bf-combined(2) 
```
You should obtain a dev accuracy of 85.1 and a test accuracy of 84.9 with the default setting.

## Reproduce our results on transfer tasks
To reproduce our results on transfer tasks, clone [SentEval](https://github.com/facebookresearch/SentEval) and set **PATH_SENTEVAL**, **PATH_TRANSFER_TASKS** in *evaluate_model.py*, then run:
```bash
python evaluate_model.py
```

Note that while BiLSTM with BF-max pooling provides good features for many different tasks, our approach also obtains strong results on STS tasks which evaluate the quality of the cosine metrics in the embedding space.
