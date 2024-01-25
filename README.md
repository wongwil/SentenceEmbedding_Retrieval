# Memory-based generation: Enabling LMs to handle long input texts by extracting useful information

As part of the course "Computational Semantics for NLP" at ETH Zurich, we are exploring different methods of enabling transformer-based LMs to handle 
long text sequences. Our project investigates the usage of sentence embeddings to address the limitations of transformer-based language models (LMs) in 
processing longer texts due to their fixed input size. In particular, in the context of solving reading questions that involve longer
articles, e.g., more than 5000 tokens, smaller language models face the challenge of truncating the text, which can lead to a loss of 
important information needed to accurately answer the question. To address this problem, we explore several retrieval methods 
that are capable of identifying relevant chunks within an article. By identifying these chunks, only a small subset of the 
text needs to be fed into the language model. This study looks at the use of sentence embeddings and compares their performance to simple
retrieval methods such as TF-IDF, random selection, and selection with preference for the beginning and end of a long text. By investigating 
retrieval methods, the project aims to improve the effectiveness of transformer-based LMs in dealing with longer texts and to provide more accurate
answers to reading questions.

<img width="786" alt="image" src="https://github.com/wongwil/SentenceEmbedding_Retrieval/assets/11984597/fd7e5034-01b2-4ae5-903f-970930899302">

Authors: Mihailo Grbic, Jakub Lucki, William Wong

For more information, please refer to our paper.

## Reproduce the experiments
To perform the experiments for the different models, run the Notebooks:
- Experiments_Deberta.ipynb
- Experiments_Longformer.ipynb
- Experiments_Roberta.ipynb

Make sure that all .py files are included in the project.

## Reproduce the analysis
To reproduce the plots used in our analysis section, run the Notebooks:
- Analysis.ipynb
