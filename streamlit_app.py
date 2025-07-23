import streamlit as st
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


st.set_page_config(page_title="Penguin Classifier", layout="wide")


st.title("🐧 Penguin Classifier 🐧")
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
df.sunheader("Случайные 10 строк")
st.datarfame(df.sample(10), use_container_width = True)
