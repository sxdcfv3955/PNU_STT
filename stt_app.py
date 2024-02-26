import numpy as np
import pandas as pd
import pickle

import streamlit as st
from io import StringIO

!pip install sentence_transformers
!pip install ko_sentence_transformers

from sentence_transformers import SentenceTransformer, models, util
from ko_sentence_transformers.models import KoBertTransformer

import time
from datetime import datetime


now = datetime.now()
start = time.time()


st.title('DATALAB STT TEST')

uploaded_file1 = st.file_uploader("Choose a file 1")
uploaded_file2 = st.file_uploader("Choose a file 2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    
    ufile1 = pd.read_csv(uploaded_file1, sep='\n', header=None, encoding="utf-8")
    ufile1.columns = ['TEXT']
    file1 = list(ufile1['TEXT'])

    ufile2 = pd.read_csv(uploaded_file2, sep='\n', header=None, encoding="utf-8")
    ufile2.columns = ['TEXT']
    file2 = list(ufile2['TEXT'])
    
    
#     graph = open("graph_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8")
#     result = open("result_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8")

    graph_ = []
    result_ = []
    
    embedder = SentenceTransformer("jhgan/ko-sbert-sts")

    corpus = file1
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    
    queries = file2
    
    if(str(uploaded_file1.name) == str(uploaded_file2.name)) : 
        st.write(str(uploaded_file1.name) + " == " + str(uploaded_file2.name))
        top_k = 7
        c1 = 0
        q = 1
        
        
        for query in queries:
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()

            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

#             print("\n\n======================\n", file = result)
#             print("Query(%i):" %(q), query + "\n", file = result)
            
            result_.append("\n\n======================\n")
            result_.append("Query(" + str(q) + "):" + str(query) + "\n")

            for idx in top_results[0:top_k]:
                if q != idx+1:
                    
#                     print("Corpus(%i): " %(idx+1) + corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]), file = result)
                    result_.append("Corpus(" + str(idx.item()+1) + "):" + str(corpus[idx].strip()) + " (Score: " + str(cos_scores[idx].item()) + " )")
    
                    if c1 == 0:
#                         print("%.4f, %i, %i" %(cos_scores[idx], q, idx), file = graph)
                        graph_.append(str(cos_scores[idx].item()) + "," +  str(q) + "," + str(idx.item()))
                        c1 += 1

            c1 = 0
            q += 1
    
    
    else : 
        st.write(str(uploaded_file1.name) + " != " + str(uploaded_file2.name))
        
        top_k = 3
        c1 = 0
        q = 1

        for query in queries:
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()

            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

#             print("\n\n======================\n", file = result)
#             print("Query(%i):" %(q), query + "\n", file = result)
            
            result_.append("\n\n======================\n")
            result_.append("Query(" + str(q) + "):" + str(query) + "\n")

            for idx in top_results[0:top_k]:
#                 print("Corpus(%i): " %(idx+1) + corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]), file = result)
                result_.append("Corpus(" + str(idx.item()+1) + "):" + str(corpus[idx].strip()) + " (Score: " + str(cos_scores[idx].item()) + " )")
                
                if c1 == 0:
#                     print("%.4f, %i, %i" %(cos_scores[idx], q, idx), file = graph)
                    graph_.append(str(cos_scores[idx].item()) + "," +  str(q) + "," + str(idx.item()))
                    c1 += 1
            c1 = 0
            q += 1
            
    graph.close()
    result.close()
    
#     result_ = pd.DataFrame(result_)
#     result_file = result_.to_csv(index=False)

#     st.download_button(
#         label="Download Result File",
#         data=result_file,
#         file_name='result.text',
#         mime='text/csv',
#     )
    
#     graph_ = pd.DataFrame(graph_)
#     graph_file = graph_.to_csv(index=False)

#     st.download_button(
#         label="Download Graph File",
#         data=graph_file,
#         file_name='graph.text',
#         mime='text/csv',
#     )
    
    if st.button('Result File Download'):
        with open("result2_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8") as r : 
            for num, i in enumerate(result_):
                if num+1 < len(result_):
                    r.write(''.join(i) + "\n")
                else:
                    r.write(''.join(i))
                    st.write("Result File Download Done")
            
    if st.button('Graph File Download'):
        with open("graph2_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8") as g : 
            for num, i in enumerate(graph_):
                if num+1 < len(graph_):
                    g.write(''.join(i) + "\n")
                else:
                    g.write(''.join(i))
                    st.write("Graph File Download Done")

                    
    
