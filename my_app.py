import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np


st.header("The big header")
st.subheader("This is my cool streamlit app")

st.title("This is my title")

st.text("this is just text")

st.write("this is also text")

# load data
df_iris = px.data.iris()


show_iris = st.checkbox("Do you want to see the thing?")

if show_iris:
    df_iris

# load model
with open("saved_iris_model.pkl", "rb") as f:
    clf = pickle.load(f)

"What's the type of clf?"
the_type = type(clf)
the_type 

# get user input
s_w = st.number_input("Sepal Width")
s_l = st.number_input("Sepal Length", value=1.)
p_w = st.number_input("Petal Width")
p_l = st.number_input("Petal Length")

user_input = np.array([s_w, s_l, p_w, p_l])

f"You entered {user_input}"

prediction = clf.predict(user_input.reshape(1, -1))
prediction

# fig = px.scatter(df_iris, x="sepal_width", y="sepal_length", color="species")
# fig

# df = px.data.gapminder()
# fig2 = px.scatter_geo(df, locations="iso_alpha", color="continent", hover_name="country", size="pop",
#                animation_frame="year", projection="natural earth")
# fig2

st.balloons()

"ool"