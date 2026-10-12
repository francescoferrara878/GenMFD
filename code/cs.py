import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize, minmax_scale


def normalize_cs(cs):
    cs = minmax_scale(cs, axis=0)
    return normalize(cs, norm='l1', axis=1)


def M_WS2file(M_WS, cs_type):
    path = os.path.join(config['summary_path'],  cs_type + '.txt')
    lex = {}
    for i, v in enumerate(M_WS):
        lex[id2str[i]] = v
    pd.DataFrame.from_dict(lex, orient='index').to_csv(path, sep='\t',
                                                       header=None)

def generate_lexicon(document_label, text_data, ngram_range=(1,1), token_pattern=r"(?u)\b\w\w+\b", cutoff=1, vocab=None):
    # Document-label matrix
    M_DE = document_label

    extractor = CountVectorizer(ngram_range=ngram_range, min_df=cutoff, max_df=1.0,
                                token_pattern=token_pattern, vocabulary=vocab)
    M_WD = extractor.fit_transform(text_data) # [n_docs, vocabulary]
    vocab = extractor.vocabulary_
    
    # Generate with RAW frequencies
    M_WS = M_WD.T.dot(M_DE)
    M_WS_raw = np.copy(M_WS)


    # Generate with NORM frequencies
    M_WD = normalize(M_WD, norm='l1', axis=1) # M_WD : [n_docs, vocab_size]
    M_WS = M_WD.T.dot(M_DE)
    M_WS_norm = np.copy(M_WS)
    
    
    # Generate with TF-IDF frequencies
    tfidf_extractor = TfidfVectorizer(vocabulary=vocab, token_pattern=token_pattern)
    M_WD = tfidf_extractor.fit_transform(text_data) # M_WD: [n_docs, vocab_size]
    M_WS = M_WD.T.dot(M_DE)
    M_WS_tfidf = np.copy(M_WS)
    
    return M_WS_raw, M_WS_norm, M_WS_tfidf, vocab, extractor

def generate_normalize_lexicon(annotations, texts, ngram_range=(1,1), cutoff=1, lemmapos=False, vocab=None):
    
    token_pattern = r"(?u)\b\w\w+#\w\b" if lemmapos else r"(?u)\b\w\w+\b"
    
    raw, norm, tfidf, vocab, extractor = generate_lexicon(annotations.values, 
                                                          texts,
                                                          ngram_range=ngram_range,
                                                          cutoff=cutoff,
                                                          token_pattern=token_pattern,
                                                          vocab=vocab)

    raw = normalize(minmax_scale(raw), axis=1, norm='l1')
    norm = normalize(minmax_scale(norm), axis=1, norm='l1')
    tfidf = normalize(minmax_scale(tfidf), axis=1, norm='l1')

    # CountVectorizer's vocabulary_ lists words in order of first occurrence, while the rows of
    # the matrices follow the column indices (alphabetical order): sort by index to keep them aligned
    vocab = pd.Series(vocab).sort_values()
    cs_raw = pd.DataFrame(index=vocab.index, data=raw, columns=annotations.columns)
    cs_norm = pd.DataFrame(index=vocab.index, data=norm, columns=annotations.columns)
    cs_tfidf = pd.DataFrame(index=vocab.index, data=tfidf, columns=annotations.columns)
    
    lexica = dict()
    lexica['raw'] = cs_raw
    lexica['norm'] = cs_norm
    lexica['tfidf'] = cs_tfidf
    lexica['vocab'] = vocab
    
    analyzer = extractor.build_analyzer()
    
    return lexica, analyzer