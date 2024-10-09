import re
import math
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

# Baixar recursos necessários do NLTK
nltk.download('stopwords')

# 1. Coleta de Documentos
documents = {
    1: "Python é uma linguagem de programação poderosa usada para desenvolvimento web, ciência de dados e muito mais.",
    2: "Java é amplamente utilizado para desenvolver aplicativos corporativos robustos.",
    3: "A ciência de dados envolve estatísticas, programação e habilidades de negócios.",
    4: "O desenvolvimento web pode ser feito usando várias linguagens, incluindo JavaScript, Python e PHP.",
    5: "Machine learning é uma aplicação da inteligência artificial que fornece aos sistemas a capacidade de aprender e melhorar automaticamente."
}

# 2. Pré-processamento dos Dados
def preprocess(text):
    # Tokenização usando RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    
    # Remoção de Stop Words
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = SnowballStemmer('portuguese')
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# 3. Construção do Índice Invertido com TF-IDF
tf = defaultdict(dict)
df = defaultdict(int)
N = len(documents)

for doc_id, text in documents.items():
    tokens = preprocess(text)
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1
    for token, count in token_counts.items():
        tf[doc_id][token] = count / len(tokens)
        df[token] += 1

inverted_index = defaultdict(dict)
for doc_id, tokens in tf.items():
    for token, tf_value in tokens.items():
        idf = math.log(N / (df[token]))
        tf_idf = tf_value * idf
        inverted_index[token][doc_id] = tf_idf

# 4. Implementação da Função de Busca
def search(query):
    tokens = preprocess(query)
    if not tokens:
        return []
    
    results = {}
    for token in tokens:
        if token in inverted_index:
            for doc_id, score in inverted_index[token].items():
                if doc_id in results:
                    results[doc_id] += score
                else:
                    results[doc_id] = score
    ranked_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return ranked_results

# 5. Testando a Ferramenta
queries = [
    "desenvolvimento web",
    "linguagem de programação",
    "inteligência artificial",
    "ciência de dados",
    "Java OR Python",
]

for query in queries:
    results = search(query)
    print(f"\nConsulta: '{query}'\nResultados:")
    if results:
        for doc_id, score in results:
            print(f"Documento {doc_id}: Score {score:.4f} - {documents[doc_id]}")
    else:
        print("Nenhum documento encontrado.")
