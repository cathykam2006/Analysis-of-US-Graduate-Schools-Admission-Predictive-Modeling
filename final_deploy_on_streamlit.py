#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import required libraries 

import streamlit as st
import pickle
import numpy as np
from sklearn import linear_model

st.write("""
# Welcome to the U.S. Master Admission Rate Predictor (by Cathy Kam)
""")

st.markdown('''This app allows you to predict your successful admission 
            rate of U.S. master's degrees. You will be evaluated based on several 
            academic parameters, namely a. GRE Scores; b. TOEFL Scores; c. University 
            Rating; d. Statement of Purpose; e. Letter of Recommendation; 
            f. Cumulative GPA and; g. Research Experience respectively. Good luck!''')

# ------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------
# Step1: Receive input from user (insert the final tested/validated set of code from deploy1)

# GRE Scores ( out of 340 ) 
# PROBLEM-2 : create a list of possible gre_scores to choose, and add the list as option in the next code. 

lst1 = list(range(100, 401))
gre_scores = st.selectbox('Select the GRE Scores that you got: ',(lst1))
st.write('You selected:', gre_scores)


lst2 = list(range(60, 121))
# TOEFL Scores ( out of 120 )
# PROBLEM-3 : create a list of possible toefl_scores to choose, and add the list as option in the next code.
toefl_scores = st.selectbox('Select the TOEFL Scores that you got: ',(lst2))
st.write('You selected:', toefl_scores)

# University Rating ( out of 5 )
uni_rating = st.slider('Select your current University Rating - 5 is the highest', 1, 5) 
st.write(uni_rating)

# Statement of Purpose ( out of 5 )
sop = st.slider('Rate your Statement of Purpose - 5 is the highest', 1, 5) 
st.write(sop)

# Letter of Recommendation ( out of 5 )
lor = st.slider('Rate your Letter of Recommendation - 5 is the highest', 1, 5) 
st.write(lor)

# Cumulative GPA ( out of 10 )
cgpa = st.slider('Select your Cumulative GPA - on a scale of 1-10', 1, 10) 
st.write(cgpa)

# Research Experience ( either 0 or 1 )
research = st.selectbox('Do you have solid research experience No - 0, Yes - 1? ',(0, 1))
st.write('You selected:', research)

# PROBLEM-1 : Change datatype of research from string to integer(yes = 1, and no = 0) 
# create a list called student with 7 items in the list, all must be positive integers

student = [gre_scores, toefl_scores, uni_rating, sop, lor, cgpa, research]
# ------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------
# step2: Run predictor (insert the final tested/validated set of code from deploy2)

#Open the saved model
filename = 'us_ad_linear1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Run the predictor 
st.subheader("Let's run the prediction!")
if st.button("Predict"):
    student = np.asarray(student)
    rescaledX_student = student.reshape(1,-1)
    student_pred = loaded_model.predict(rescaledX_student)

    if student_pred[0] >= 0.72:
        st.write("Congratulations! Your admission rate is >= 0.72/1. You're very likely to be successful!")
    elif student_pred[0] >= 0.64 < 0.72:
        st.write("Good! Your admission rate is between 0.64-0.72/1. You have a decent chance.")
    else:
        st.write("Emm...Your estimated admission rate is <0.64/1. Maybe you need to work harder to secure the admission.")
# -------------------------------------------------------------------------------------


# In[6]:





# In[ ]:




