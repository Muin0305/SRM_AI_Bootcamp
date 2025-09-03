import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="🕊 Ceasefire Success Predictor", layout="wide")

# Title and description
st.title("🕊 Ceasefire Success Predictor")
st.markdown("""
Этот дашборд анализирует успешность перемирий в вооруженных конфликтах. 
Вы можете выбрать модель, настроить параметры и предсказать успех перемирия на основе введенных характеристик.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        logger.info("Loading ceasefire dataset...")
        df = pd.read_csv('ceasefires_dataset.csv')
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        columns_to_drop = [
            'link_id1', 'link_id2', 'link_id3', 'link_id4', 'link_id5', 'link_id6', 'link_id7', 'link_id8',
            'link_id9', 'link_id10', 'link_id11', 'link_id12', 'link_id13', 'link_id14', 'link_id15', 'link_id16',
            'link_id17', 'link_id18', 'link_id19', 'link_id20', 'link_id21', 'link_id22', 'link_id23', 'link_id24',
            'link_id25', 'link_id26', 'link_id27', 'link_id28', 'link_id29', 'link_id30', 'link_id31', 'link_id32',
            'link_id33', 'link_id34', 'link_id35', 'evidence_onset', 'evidence_end', 'comment', 'factivia_source',
            'cc', 'cf_id', 'uniq_id', 'ucdp_actor_id', 'actor_name', 'ucdp_acd_id', 'ucdp_dyad', 'pax_id',
            'cf_dec_yr', 'cf_dec_month', 'cf_dec_day', 'cf_effect_yr', 'cf_effect_month', 'cf_effect_day',
            'p_other_comment', 'cf_pp', 'splinter', 'end_yr', 'end_month', 'end_day', 'factivia_page', 'truce',
            'recentadditions', 'coder', 'id'
        ]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        df['success'] = df['ended'].apply(lambda x: 1 if x in [1, 4] else 0)
        df = df.drop('ended', axis=1)

        # Process fixed_time
        def clean_fixed_time(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x).lower()
            numbers = re.findall(r'\d+', x_str)
            return float(numbers[0]) if numbers else np.nan

        df['fixed_time'] = df['fixed_time'].apply(clean_fixed_time)
        df['is_fixed_time_unclear'] = df['fixed_time'].isna().astype(int)

        # Process mediator columns
        categorical_cols = ['location', 'region', 'link', 'side', 'partial', 'written', 'fixed', 'nsa_frac',
                           'p_humanitarian', 'p_peaceprocess', 'p_holiday', 'p_election', 'p_other', 'p_unclear',
                           'ceasefire_class', 'timing', 'implement', 'enforcement', 'ddr', 'is_fixed_time_unclear']
        numeric_cols = ['fixed_time']

        if 'mediator_nego' in df.columns:
            df['mediator_nego_count'] = df['mediator_nego'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
            df['has_mediator_nego'] = df['mediator_nego'].notna().astype(int)
            categorical_cols.append('has_mediator_nego')
            numeric_cols.append('mediator_nego_count')
            df = df.drop('mediator_nego', axis=1)

        if 'mediator_send' in df.columns:
            df['mediator_send_count'] = df['mediator_send'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
            df['has_mediator_send'] = df['mediator_send'].notna().astype(int)
            categorical_cols.append('has_mediator_send')
            numeric_cols.append('mediator_send_count')
            df = df.drop('mediator_send', axis=1)

        logger.info("Data preprocessing completed")
        return df, categorical_cols, numeric_cols
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None, None, None

# Load data
df, categorical_cols, numeric_cols = load_data()
if df is None:
    st.stop()

# Display dataset
st.write("## 🧾 Датасет")
st.dataframe(df.sample(10), use_container_width=True)

# Data visualizations
st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)
with col1:
    fig_region = px.histogram(df, x="region", color="success", barmode="group",
                              title="Перемирия по регионам и успешности")
    st.plotly_chart(fig_region, use_container_width=True)
with col2:
    fig_written = px.histogram(df, x="written", color="success", barmode="group",
                               title="Успешность перемирий по наличию письменного соглашения")
    st.plotly_chart(fig_written, use_container_width=True)

fig_box = px.box(df, x="success", y="fixed_time", color="is_fixed_time_unclear",
                 title="Распределение длительности перемирия по успешности")
st.plotly_chart(fig_box, use_container_width=True)

# Preprocessing pipeline
try:
    transformers = []
    if numeric_cols:
        transformers.append(('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Split data
    X = df.drop('success', axis=1)
    y = df['success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
except Exception as e:
    logger.error(f"Error in preprocessing: {str(e)}")
    st.error(f"Ошибка в предобработке данных: {str(e)}")
    st.stop()

# Sidebar for model settings
st.sidebar.header("⚙️ Настройки модели")
model_name = st.sidebar.selectbox("Выберите модель", ["Logistic Regression", "Random Forest"])

if model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)
    max_depth = st.sidebar.slider("max_depth", 2, 20, 5)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
else:
    model = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', max_iter=1000)

# Train model
try:
    logger.info("Training model...")
    model.fit(X_train_processed, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_processed))
    test_acc = accuracy_score(y_test, model.predict(X_test_processed))
    train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train_processed)[:, 1])
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])
    logger.info("Model training completed")
except Exception as e:
    logger.error(f"Error training model: {str(e)}")
    st.error(f"Ошибка при обучении модели: {str(e)}")
    st.stop()

# Display model results
st.subheader("📈 Результаты модели")
st.write(f"**Точность на обучающей выборке:** {train_acc:.2f}")
st.write(f"**Точность на тестовой выборке:** {test_acc:.2f}")
st.write(f"**ROC AUC на обучающей выборке:** {train_roc_auc:.2f}")
st.write(f"**ROC AUC на тестовой выборке:** {test_roc_auc:.2f}")

# Feature importance for Random Forest
if model_name == "Random Forest":
    try:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Признак': feature_names, 'Значимость': importances})
        fig_imp = px.bar(feat_df.sort_values('Значимость', ascending=False).head(10), x='Значимость', y='Признак', orientation='h',
                         title="📊 Важность признаков")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        logger.error(f"Error displaying feature importance: {str(e)}")
        st.error(f"Ошибка при отображении важности признаков: {str(e)}")

# Sidebar for user input
st.sidebar.header("🔍 Ввод параметров перемирия")
region = st.sidebar.selectbox("Регион", df['region'].unique())
written = st.sidebar.selectbox("Письменное соглашение", df['written'].unique())
fixed_time = st.sidebar.slider("Длительность перемирия (дни)", 0.0, float(df['fixed_time'].max()), float(df['fixed_time'].median()))
is_fixed_time_unclear = st.sidebar.selectbox("Неопределенность длительности", [0, 1])
side = st.sidebar.selectbox("Сторона", df['side'].unique())
partial = st.sidebar.selectbox("Частичное перемирие", df['partial'].unique())
ceasefire_class = st.sidebar.selectbox("Класс перемирия", df['ceasefire_class'].unique())

# Create user input DataFrame
try:
    user_input = pd.DataFrame([{
        'region': region,
        'written': written,
        'fixed_time': fixed_time,
        'is_fixed_time_unclear': is_fixed_time_unclear,
        'side': side,
        'partial': partial,
        'ceasefire_class': ceasefire_class,
        'location': df['location'].mode()[0],
        'link': df['link'].mode()[0],
        'fixed': df['fixed'].mode()[0],
        'nsa_frac': df['nsa_frac'].mode()[0],
        'p_humanitarian': df['p_humanitarian'].mode()[0],
        'p_peaceprocess': df['p_peaceprocess'].mode()[0],
        'p_holiday': df['p_holiday'].mode()[0],
        'p_election': df['p_election'].mode()[0],
        'p_other': df['p_other'].mode()[0],
        'p_unclear': df['p_unclear'].mode()[0],
        'timing': df['timing'].mode()[0],
        'implement': df['implement'].mode()[0],
        'enforcement': df['enforcement'].mode()[0],
        'ddr': df['ddr'].mode()[0]
    }])

    if 'mediator_nego_count' in df.columns:
        user_input['mediator_nego_count'] = df['mediator_nego_count'].median()
        user_input['has_mediator_nego'] = df['has_mediator_nego'].mode()[0]
    if 'mediator_send_count' in df.columns:
        user_input['mediator_send_count'] = df['mediator_send_count'].median()
        user_input['has_mediator_send'] = df['has_mediator_send'].mode()[0]
except Exception as e:
    logger.error(f"Error creating user input: {str(e)}")
    st.error(f"Ошибка при создании пользовательского ввода: {str(e)}")
    st.stop()

# Predict
if st.sidebar.button("🔮 Предсказать"):
    try:
        user_processed = preprocessor.transform(user_input)
        prediction = model.predict(user_processed)[0]
        proba = model.predict_proba(user_processed)[0]
        st.sidebar.markdown(f"### 🧠 Вероятность успеха перемирия: **{proba[1]*100:.1f}%**")
        st.sidebar.markdown(f"**Модель прогнозирует:** {'✅ Успешно' if prediction == 1 else '❌ Неуспешно'}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        st.error(f"Ошибка при предсказании: {str(e)}")
