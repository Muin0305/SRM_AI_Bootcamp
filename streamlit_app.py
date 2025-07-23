import streamlit as st
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import TargetEncoder
import plotly.express as px


st.set_page_config(page_title="🧊 Penguin Classifier", layout="wide")
st.title("🧊 Penguin Classifier - Обучение и предсказание")
st.write("## Работа с датасетом пингвинов")

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

st.subheader("📋 Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x="species", color="island", barmode="group",
                        title="Распределение видов по островам")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(df, x="bill_length_mm", y="flipper_length_mm", color="species",
                      title="Длина клюва vs Длина крыла")
    st.plotly_chart(fig2, use_container_width=True)
X = df.drop(['species'], axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

encoder = ce.TargetEncoder(cols=['island', 'sex'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train_encoded, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train_encoded))
    acc_test = accuracy_score(y_test, model.predict(X_test_encoded))
    results.append({
        'Model': name,
        'Train Accuracy': round(acc_train, 2),
        'Test Accuracy': round(acc_test, 2)
    })

st.write("### 📊 Сравнение моделей по точности")
st.table(pd.DataFrame(results))
