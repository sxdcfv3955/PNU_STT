import numpy as np
import pandas as pd
import pickle

import streamlit as st
from io import StringIO

from sentence_transformers import SentenceTransformer, models, util
from ko_sentence_transformers.models import KoBertTransformer

from re import split
import matplotlib.pyplot as plt    #맷플롯립의 pyplot 모듈
import numpy as np

import time
from datetime import datetime

embedder = SentenceTransformer("jhgan/ko-sbert-sts")

now = datetime.now()
start = time.time()


st.title('DATALAB STT TEST')

st.write("---")

uploaded_file1 = st.file_uploader("Choose a file 1")
uploaded_file2 = st.file_uploader("Choose a file 2")

if uploaded_file1 is not None and uploaded_file2 is not None:
    
#     ufile1 = pd.read_csv(uploaded_file1, sep='\n', header=None, encoding="utf-8")
#     ufile1.columns = ['TEXT']
#     file1 = list(ufile1['TEXT'])

#     ufile2 = pd.read_csv(uploaded_file2, sep='\n', header=None, encoding="utf-8")
#     ufile2.columns = ['TEXT']
#     file2 = list(ufile2['TEXT'])


    # file1=[]
    # with open(uploaded_file1.name, "r", encoding="utf-8") as f1:
    #     for i in f1:
    #         file1.append(i.strip())
            
    # file2=[]
    # with open(uploaded_file2.name, "r", encoding="utf-8") as f2:
    #     for i in f2:
    #         file2.append(i.strip())

    
    file1=[]
    ufile1 = StringIO(uploaded_file1.getvalue().decode("utf-8"))

    for i in ufile1 :
        file1.append(i)
            
    file2=[]
    ufile2 = StringIO(uploaded_file2.getvalue().decode("utf-8"))

    for i in ufile2 :
        file2.append(i)
    
    
#     graph = open("graph_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8")
#     result = open("result_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8")

    graph_ = ""
    result_ = ""

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
            
            result_+=("\n\n======================\n")
            result_+=("Query(" + str(q) + "):" + str(query) + "\n")
            result_+='\n'

            for idx in top_results[0:top_k]:
                if q != idx+1:
                    
#                     print("Corpus(%i): " %(idx+1) + corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]), file = result)
                    result_+=("Corpus(" + str(idx.item()+1) + "):" + str(corpus[idx].strip()) + " (Score: " + str(round(cos_scores[idx].item(),2)) + " )")
                    result_+='\n'

                    if c1 == 0:
#                         print("%.4f, %i, %i" %(cos_scores[idx], q, idx), file = graph)
                        graph_+=(str(round(cos_scores[idx].item(),2)) + "," +  str(q) + "," + str(idx.item()+1))
                        graph_+='\n'
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
            
            result_+=("\n\n======================\n")
            result_+=("Query(" + str(q) + "):" + str(query) + "\n")
            result_+='\n'
            for idx in top_results[0:top_k]:
#                 print("Corpus(%i): " %(idx+1) + corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]), file = result)
                result_+=("Corpus(" + str(idx.item()+1) + "):" + str(corpus[idx].strip()) + " (Score: " + str(round(cos_scores[idx].item(),2)) + " )")
                result_+='\n'
                
                if c1 == 0:
#                     print("%.4f, %i, %i" %(cos_scores[idx], q, idx), file = graph)
                    graph_+=(str(round(cos_scores[idx].item(),2)) + "," +  str(q) + "," + str(idx.item()+1))
                    graph_+='\n'

                    c1 += 1
            c1 = 0
            q += 1
                
#     graph.close()
#     result.close()

    imsi1 = []
    a1 = []
    b1 = []
    c1 = []

    content1 = graph_.split('\n')[:-1]
    print(content1[0])
    
    i = 0
    for line in content1:
        imsi1 = content1[i].split(',')
        print(imsi1)
        a1.append(float(imsi1[0]))
        b1.append(float(imsi1[1]))
        c1.append(float(imsi1[2]))
        i += 1
    
    fig, ax = plt.subplots(figsize=(18,6))
    # x축에는 query 순서값, y축에는 sbert score값을 표시한다.
    plt.plot(b1, a1, color = 'red', marker = 'o', linestyle = 'solid', label='Sentence BERT')
    plt.axhline(0.75, 0.01, 0.99, color='blue', linestyle='--', linewidth=1)
    # plt.hlines(0.7, 1.0, 66.0, color='green', linestyle='--', linewidth=1) # solid
    plt.xticks(np.arange(1, i+1, 1))
    plt.yticks(np.arange(0.3, 1.2, 0.1))
    plt.legend()

    # 제목을 설정
    plt.title('Top similar sentence in corpus') # corpus(rfp01)중에서 query(rfp01)와 가장 유사한 문장

    plt.ylabel('Score')
    plt.xlabel('No. of Query')
    plt.savefig('graph_Result.png', dpi = 1200)

    
    st.write(str(time.time()-start)+" sec")
    st.download_button('Download Result File', result_, file_name="result_"+now.strftime('%Y%m%d%H%M')+".txt")

    with open("graph_Result.png", "rb") as file:
        btn = st.download_button(
                label="Download Graph image",
                data=file,
                file_name="graph_"+now.strftime('%Y%m%d%H%M')+".png",
                mime="image/png"
              )


    # st.download_button('Download Graph File', graph_, file_name="graph_"+now.strftime('%Y%m%d%H%M')+".txt")

    
    # if st.button('Result File Download'):
    #     with open("result2_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8") as r : 
    #         for num, i in enumerate(result_):
    #             if num+1 < len(result_):
    #                 r.write(''.join(i) + "\n")
    #             else:
    #                 r.write(''.join(i))
    #                 st.write("Result File Download Done")
            
    # if st.button('Graph File Download'):
    #     with open("graph2_"+now.strftime('%Y%m%d%H%M')+".txt", 'w', encoding="utf-8") as g : 
    #         for num, i in enumerate(graph_):
    #             if num+1 < len(graph_):
    #                 g.write(''.join(i) + "\n")
    #             else:
    #                 g.write(''.join(i))
    #                 st.write("Graph File Download Done")

                    
    
