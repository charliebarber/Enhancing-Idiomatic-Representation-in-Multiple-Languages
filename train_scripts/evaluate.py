import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoConfig
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

from help_funcs import get_result

def main():
  output_dir = "/cs/student/projects1/2021/cbarber/Enhancing-Idiomatic-Representation-in-Multiple-Languages"
  model_path = output_dir + "/model/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/0/checkpoint-2880" 
  word_embedding_model = models.Transformer(model_path)
  # Create a pooling model
  pooling_model = models.Pooling(
      word_embedding_model.get_word_embedding_dimension(),
      pooling_mode_mean_tokens=True
  )

  # Combine them into a SentenceTransformer model
  model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
  
  result = get_result(path = "./model/sentence-transformers/paraphrase-multilingual-mpnet-base-v2", model = model, mode = 'dev', languages = ['EN', 'PT', 'GL'], if_tokenize = True, gen_result=True)
  print(result)


if __name__ == '__main__':
  main()