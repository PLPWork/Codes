#Core pkgs
import streamlit as st
from keybert import KeyBERT
#from samples import texts


#NLP Pkgs
import spacy_streamlit
import spacy
import pandas as pd
nlp=spacy.load("en_core_web_sm")

import pandas as pd
import numpy as np
import spacy
import re
import nltk
import docx2txt
nltk.download('stopwords')

@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=True)
def load_model():
    model = KeyBERT("distilbert-base-nli-mean-tokens")
    return model
model = load_model()

def main():
    """A simple nlp app"""
    st.title("NLP As a Service")
    menu=["Home","NER","Keyword_extraction","Resume_summarizer","Resume_matching","Resume_personality_insights","Core_details"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Home":
        st.subheader("Tokenization")
        raw_text=st.text_area("Your Text","Enter your text here")
        docx=nlp(raw_text)
        if st.button("Tokenize"):
            spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_'])
    elif choice=="NER":
        st.subheader("Named Entity recognition")
        raw_text=st.text_area("Your Text","Enter your text here")
        docx=nlp(raw_text)
        if st.button("Tokenize"):
            spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
    elif choice=="Keyword_extraction":
        st.subheader("Keyword_extraction")
        placeholder = st.empty()
        text_input = placeholder.text_area("Type in some text you want to analyze", height=300)
        top_n = st.sidebar.slider("Select number of keywords to extract", 5, 20, 10, 1)
        min_ngram = st.sidebar.number_input("Min ngram", 1, 5, 1, 1)
        max_ngram = st.sidebar.number_input("Max ngram", min_ngram, 5, 3, step=1)
        st.sidebar.code(f"ngram_range = ({min_ngram}, {max_ngram})")
        params = {
        "docs": text_input,
        "top_n": top_n,
        "keyphrase_ngram_range": (min_ngram, max_ngram),
        "stop_words": "english",
        }
        if st.button("Extract"):
            keywords = model.extract_keywords(**params)
            if keywords != []:
                st.info("Extracted keywords")
                keywords = pd.DataFrame(keywords, columns=["keyword", "relevance"])
                st.table(keywords)
    elif choice=="Core_details":
        st.subheader("Core_details")
        data_file=st.file_uploader("Upload CSV",type=["csv"])
        if data_file is not None:
            data=pd.read_csv(data_file)
            data=data.apply(str)
            #Extracting experience
            
            def extract_education(job_desc):
                regex1 = r'[\w]*[\’s]*?[\'s]*?\s*degree?' 
                regex2 = r'[\w]*[\’s]*?[\'s]*?\s*[\w]*\s*[\w]*[\'s]*\s*degree?'

                regexList = [regex1, regex2]

                found_regex_list = []

                for x in regexList:
                    if any(re.findall(x, data)):
                        some_list = re.findall(x, data)     
                        found_regex_list.append(some_list)
  

                return set([item.strip() for sublist in found_regex_list for item in sublist])
            education = extract_education(data)
            st.table(education)
        
            

    elif choice=="Resume_matching":
        st.subheader("Core_details")
        job_desc=st.file_uploader("Upload doc",type=["docx"])
        resume=st.file_uploader("Upload docx",type=["docx"])
        if job_desc is not None:
            raw_text = docx2txt.process(job_desc)
        if resume is not None:
            raw_text_resume = docx2txt.process(resume)
            text=[raw_text,raw_text_resume]
        if st.button("Check similarity"):
            from sklearn.feature_extraction.text import CountVectorizer
            cv=CountVectorizer()
            count_matrx=cv.fit_transform(text)
            from sklearn.metrics.pairwise import cosine_similarity
            per=cosine_similarity(count_matrx)[0][1]*100
            st.text(per)





    
if __name__=='__main__':
    main()