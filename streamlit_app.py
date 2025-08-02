import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import category_encoders as ce

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("Titanic Survival Predictor")

df = pd.read_csv("https://raw.githubusercontent.com/Muin0305/SRM_AI_Bootcamp/master/Titanic.csv")
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].replace('unknown', pd.NA)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

st.write("##Датасет")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("Визуализация данных")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.histogram(df, x="Pclass", color="Sex", barmode="group",
                                  title="Пассажиры по классам и полу"), use_container_width=True)
with col2:
    st.plotly_chart(px.histogram(df, x="Survived", color="Sex", barmode="group",
                                  title="Выживаемость по полу"), use_container_width=True)


fig = px.box(df, x="Survived", y="Age", color="Sex", title="Распределение возраста по полу и выживанию")
st.plotly_chart(fig, use_container_width=True)

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
encoder = ce.TargetEncoder(cols=['Sex', 'Embarked'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)


st.sidebar.header("Настройки модели")
model_name = st.sidebar.selectbox("Выберите модель", ["Random Forest", "Logistic Regression"])

if model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)  
    max_depth = st.sidebar.slider("max_depth", 2, 20, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train_encoded, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train_encoded))
test_acc = accuracy_score(y_test, model.predict(X_test_encoded))

st.subheader("Результаты модели")
st.write(f"**Train Accuracy:** {train_acc:.2f}")
st.write(f"**Test Accuracy:** {test_acc:.2f}")

if model_name == "Random Forest":
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Признак': X_train_encoded.columns, 'Значимость': importances})
    fig_imp = px.bar(feat_df.sort_values('Значимость'), x='Значимость', y='Признак', orientation='h',
                     title="Важность признаков")
    st.plotly_chart(fig_imp, use_container_width=True)


st.sidebar.header("Ввод параметров пассажира")
sex = st.sidebar.selectbox("Пол", df['Sex'].unique())
pclass = st.sidebar.selectbox("Класс", sorted(df['Pclass'].unique()))
age = st.sidebar.slider("Возраст", float(df['Age'].min()), float(df['Age'].max()), float(df['Age'].mean()))
fare = st.sidebar.slider("Стоимость билета", float(df['Fare'].min()), float(df['Fare'].max()), float(df['Fare'].mean()))
sibsp = st.sidebar.slider("SibSp (родственники)", 0, int(df['SibSp'].max()), 0)
parch = st.sidebar.slider("Parch (дети/родители)", 0, int(df['Parch'].max()), 0)
embarked = st.sidebar.selectbox("Порт посадки", df['Embarked'].unique())

user_input = pd.DataFrame([{
    'Sex': sex,
    'Pclass': pclass,
    'Age': age,
    'Fare': fare,
    'SibSp': sibsp,
    'Parch': parch,
    'Embarked': embarked
}])
user_encoded = encoder.transform(user_input)
user_encoded = user_encoded[X_train_encoded.columns]

if st.sidebar.button("Предсказать"):
    prediction = model.predict(user_encoded)[0]
    proba = model.predict_proba(user_encoded)[0]
    st.sidebar.markdown(f"### Вероятность выживания: **{proba[1]*100:.1f}%**")
    st.sidebar.markdown(f"**Модель прогнозирует:** {' Выжил' if prediction == 1 else 'Не выжил'}")
