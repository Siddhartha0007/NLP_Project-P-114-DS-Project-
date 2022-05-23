# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:46:53 2022

@author: Siddhartha Sarkar
"""

# Model Deployment Syntax
import streamlit as st 
import streamlit.components.v1 as stc
from sklearn.linear_model import LogisticRegression
import spacy
import pickle
import random
import docs
from spacy import displacy
import docx
from spacy.lang.en.stop_words import  STOP_WORDS
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import doc
from PyPDF2 import PdfFileReader
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from spacy.matcher import Matcher
#nltk.download('stopwords')
#nltk.download('wordnet')
import pickle
from pickle import dump
from pickle import load
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nlp = spacy.load('en_core_web_trf')
model = load(open('Random_Forest_model_intelligence.pkl','rb'))

st.title('Model Deployment: Document Classification')

st.header('User Input Resume')
st.sidebar.subheader('File_Description')

def readtxt(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

# Defining the SkillSet
skill_set=[  't-sql', 'sas', 'r', 'python', 'mariadb',
            'msexcel', 'tableau', 'xml', 'xslt', 'eib',
           'oracle', 'peoplesoft', 'sql', 'hcm', 'fcm',
           'msbi', 'html', 'css3', 
           'xml', 'javascript', 'json', 'react js', 'node.js']



#Skill Set Extraction Function

def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    
     
    
    
    skills = skill_set
    
    skillset = []
    
    # check for one-grams
    for token in tokens:
        if token in skills:
            skillset.append(token)
    
    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def uniquify(string):
    output = []
    seen = set()
    for word in string.split():
        if word not in seen:
            output.append(word)
            seen.add(word)
    return ' '.join(output)


def user_input():
    
    docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
    if st.button("Process"):
    	if docx_file is not None:
            
        	file_details = {"Filename":docx_file.name,
                    "FileType":docx_file.type,"FileSize":docx_file.size}
	       	st.sidebar.write(file_details)
    raw_text = readtxt(docx_file) # Parse in the uploadFile Class directory
    li=[]
    li.append(raw_text)
    
    data = {'resume':li}
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input()


clean = []
lz = WordNetLemmatizer()
for i in range(df.shape[0]):
    review = re.sub(
        '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
        " ",
        df["resume"].iloc[i],
    )
    review = re.sub(r"[0-9]+", " ", review) # Remove Numbers
    review = review.lower()
    review = review.split()
    lm = WordNetLemmatizer()
    review = [ lz.lemmatize(word) for word in review if word not in STOP_WORDS]
    review = " ".join(review)
    clean.append(review)


df["Clean_Resume"] = clean
df["Clean_Resume"]=df["Clean_Resume"].apply(uniquify)

st.write(df["Clean_Resume"])
df["Skills"]=df["Clean_Resume"].apply(extract_skills)

st.subheader('Skills Extracted From Resume')
st.write(df["Skills"])

# Creating Resume Data Frame
resume_data=pd.DataFrame()
resume_data["Resume"]=df["Clean_Resume"]

#  Vectorisation


#requiredText = df.Clean_Resume
#word_vectorizer = TfidfVectorizer(analyzer='word',
                                  #stop_words='english',max_features=1500)
#word_vectorizer.fit(requiredText)
#WordFeatures = word_vectorizer.fit_transform(requiredText)

#  Name Entity Recognisation
#text1=nlp(df["Clean_Resume"][0])
#dis=displacy.render(text1, style = "ent")
#st.sidebar.subheader('Name Entity Recognisation')
#st.sidebar.write(dis)
# Model Prediction
y_pred = model.predict(df["Clean_Resume"])
st.subheader('Model Prediction')
st.write(y_pred)
from sklearn.preprocessing import LabelEncoder
le_encoder=LabelEncoder()

if __name__ == '__user_input__':
	user_input()