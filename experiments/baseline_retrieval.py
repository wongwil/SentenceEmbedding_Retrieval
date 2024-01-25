from numpy import dot
from numpy.linalg import norm
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import pickle

def split_text_into_sentences(text):
  text = text.replace("\n", "") # replace linebreaks
  sentences = text.split(". ") # split sentences
  sentences = [string for string in sentences if string] # remove empty strings ""
  sentences = sentences[:-1] if not sentences[-1].strip() else sentences # make sure last sentece is not empty
  sentences = [sentence if sentence.endswith(".") else sentence + ". " for sentence in sentences] # last sentence usually just ends with "." instead of ". ", do not add delimiter for them
  return sentences

def split_text_into_sentece_clusters(text, idx, cluster_file = "sentence_clusters.pkl"):
  with open(cluster_file, "rb") as fp:
    clusters = pickle.load(fp)

  cluster = clusters[idx]
  sentences = split_text_into_sentences(text)
  sentence_clusters = []
  
  current_cluster = []
  for c_idx in range(len(cluster)-1):
    if cluster[c_idx] == cluster[c_idx+1]:
      current_cluster.append(sentences[c_idx])
    else:
      current_cluster.append(sentences[c_idx])
      sentence_clusters.append(current_cluster)
      current_cluster = []
  
  current_cluster.append(sentences[len(cluster)-1])
  sentence_clusters.append(current_cluster)

  return ["".join(cluster) for cluster in sentence_clusters]

def split_text_into_chunks(text, max_token_size, idx = None):
    
    # support spliting into sentences as well
    if max_token_size == "sentence":
       return split_text_into_sentences(text)
    elif max_token_size == "sentence_cluster":
       return split_text_into_sentece_clusters(text, idx)
    elif isinstance(max_token_size, str):
      raise TypeError("Only integers or 'sentence' or 'sentence_cluster' allowed")

    text = text.replace("\n", "") # replace linebreaks
    sentences = text.split(". ") # split sentences
    sentences = [string for string in sentences if string] # remove empty strings ""
    sentences = [sentence if sentence.endswith(".") else sentence + ". " for sentence in sentences] # last sentence usually just ends with "." instead of ". ", do not add delimiter for them

    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for sentence in sentences:
        sentence_size =  len(sentence.split())
        if current_chunk_size +sentence_size < max_token_size:
            current_chunk += sentence
            current_chunk_size += sentence_size
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_chunk_size = sentence_size

    if current_chunk: # add last element
        chunks.append(current_chunk)

    return chunks

def random_sentence_cut(article, tokenizer, MAX_TOKENS=512, extra_length = 0, chunk_size = 256, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length

  if "idx" in kwargs:
    idx = kwargs["idx"]
  else:
    idx = None
  sentences = split_text_into_chunks(article, chunk_size, idx)

  # get the permutation of the sentences
  num_sentences = len(sentences)
  sentence_list = list(range(num_sentences))
  random.shuffle(sentence_list)

  selected_sentences = []
  total_tokens = 0

  # get a cut of senteces that is MAX_TOKENS long or less
  for sentence_idx in sentence_list:
    tokens = tokenizer.tokenize(sentences[sentence_idx])
    num_tokens = len(tokens)
    if total_tokens == MAX_TOKENS:
      break
    elif (total_tokens + num_tokens) <= MAX_TOKENS:
      selected_sentences.append(sentence_idx)
      total_tokens += num_tokens

  # use the senteces in the original order
  selected_sentences.sort()
  final_sentences = [sentences[i] for i in selected_sentences]

  return "".join(final_sentences), selected_sentences

def start_ending_biased_sentece_cut(article, tokenizer, MAX_TOKENS=512, extra_length = 0, chunk_size = 256, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length
  if "idx" in kwargs:
    idx = kwargs["idx"]
  else:
    idx = None
  sentences = split_text_into_chunks(article, chunk_size, idx)
  num_sentences = len(sentences)
  sentence_list = list(range(num_sentences))

  # get probability distribution for the senteces which is biased towards the start and end of the article
  quadratic = lambda x : 0.1*(x - (num_sentences/2))**2 + 1 # strictly positive quadratic function with minimum at the middle of the article
  sentence_probs = np.array([quadratic(i) for i in range(num_sentences)])
  sentence_probs = sentence_probs/np.sum(sentence_probs)

  sentence_list = np.random.choice(sentence_list, size=num_sentences, replace=False, p=sentence_probs)

  selected_sentences = []
  total_tokens = 0

  for sentence_idx in sentence_list:
    tokens = tokenizer.tokenize(sentences[sentence_idx])
    num_tokens = len(tokens)
    if total_tokens == MAX_TOKENS:
      break
    elif (total_tokens + num_tokens) <= MAX_TOKENS:
      selected_sentences.append(sentence_idx)
      total_tokens += num_tokens

  # use the senteces in the original order
  selected_sentences.sort()
  final_sentences = [sentences[i] for i in selected_sentences]

  return "".join(final_sentences), selected_sentences

def tf_idf_sentece_cut(article, tokenizer, query, MAX_TOKENS = 512, extra_length = 0, chunk_size = 256, *args, **kwargs):

  MAX_TOKENS = MAX_TOKENS - extra_length
  if "idx" in kwargs:
    idx = kwargs["idx"]
  else:
    idx = None
  sentences = split_text_into_chunks(article, chunk_size, idx)
  num_sentences = len(sentences)

  # tf_idf
  vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(4,6))
  tf_idf = vectorizer.fit_transform(sentences)
  query_vector = vectorizer.transform([query])
  cos_sim = lambda a,b : dot(a, b)/(norm(a)*norm(b))
  cosine_similarities = np.array([cos_sim(tf_idf[i].toarray().flatten(), query_vector.toarray().flatten()) for i in range(num_sentences)])
  sentence_list = np.argsort(cosine_similarities)[::-1]

  assert len(sentence_list) == num_sentences

  selected_sentences = []
  total_tokens = 0

  # get the closest sentences to tf_idf
  for sentence_idx in sentence_list:
    tokens = tokenizer.tokenize(sentences[sentence_idx] + ".")
    num_tokens = len(tokens)
    if total_tokens == MAX_TOKENS:
      break
    elif (total_tokens + num_tokens) <= MAX_TOKENS:
      selected_sentences.append(sentence_idx)
      total_tokens += num_tokens
    else:
      break

  # use the senteces in the original order
  selected_sentences.sort()
  final_sentences = [sentences[i] for i in selected_sentences]

  return "".join(final_sentences), selected_sentences

def sentence_embedding_cut(article, tokenizer, query, MAX_TOKENS = 512, extra_length = 0, chunk_size = 256, sentembb_model = None, *args, **kwargs):
    MAX_TOKENS = MAX_TOKENS - extra_length

    if "idx" in kwargs:
      idx = kwargs["idx"]
    else:
      idx = None
    sentences = split_text_into_chunks(article, chunk_size, idx)

    query_embedding = sentembb_model.encode(query)

    corpus_embeddings = sentembb_model.encode(sentences)

    similarity_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    result = list(zip(range(0, len(sentences)), similarity_scores))

    # sort them by similarity score
    sentences_sortby_similarity = sorted(result, key=lambda x: x[1], reverse=True)

    selected_sentences = []
    total_tokens = 0

    for (sentence_idx, similarity) in sentences_sortby_similarity:
        tokens = tokenizer.tokenize(sentences[sentence_idx])
        num_tokens = len(tokens)
        if total_tokens == MAX_TOKENS:
            break
        elif (total_tokens + num_tokens) <= MAX_TOKENS:
            selected_sentences.append(sentence_idx)
            total_tokens += num_tokens
        else:
            break

    # use the senteces in the original order
    selected_sentences.sort()
    final_sentences = [sentences[i] for i in selected_sentences]
    return "".join(final_sentences), selected_sentences