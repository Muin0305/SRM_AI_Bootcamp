```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve
import re

# Set page configuration
st.set_page_config(page_title="Ceasefire Success Analysis", layout="wide")

# Title and description
st.title("Анализ успешности перемирий")
st.markdown("""
Этот дашборд представляет анализ факторов, влияющих на успешность перемирий в вооруженных конфликтах. 
Использованы модели машинного обучения для предсказания успеха перемирия на основе характеристик, 
таких как регион, наличие посредников и тип соглашения.
""")

# Load data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ceasefires_dataset.csv')
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

        # Define categorical and numeric columns
        categorical_cols = ['location', 'region', 'link', 'side', 'partial', 'written', 'fixed', 'nsa_frac',
                           'p_humanitarian', 'p_peaceprocess', 'p_holiday', 'p_election', 'p_other', 'p_unclear',
                           'ceasefire_class', 'timing', 'implement', 'enforcement', 'ddr', 'is_fixed_time_unclear']
        numeric_cols = ['fixed_time']

        # Process mediator columns
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

        return df, categorical_cols, numeric_cols
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None, None, None

# Train models
@st.cache_resource
def train_models(df, categorical_cols, numeric_cols):
    try:
        X = df.drop('success', axis=1)
        y = df['success']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

        models = {
            'LogisticRegression': {
                'model': LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear'),
                'params': {'classifier__C': [0.1, 1, 10]}
            },
            'RandomForest': {
                'model': RandomForestClassifier(class_weight='balanced', random_state=42, min_samples_split=5),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [5, 10]}
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1], 'classifier__max_depth': [3, 5]}
            },
            'XGBoost': {
                'model': XGBClassifier(eval_metric='logloss', scale_pos_weight=1.2 * (y_train.value_counts()[0] / y_train.value_counts()[1]), max_depth=5, min_child_weight=2, gamma=0.2),
                'params': {'classifier__n_estimators': [100], 'classifier__learning_rate': [0.01, 0.05]}
            }
        }

        results = {}
        best_model = None
        best_auc = 0
        best_params = {}

        for name, config in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', config['model'])
            ])
            
            grid = GridSearchCV(pipeline, config['params'], cv=StratifiedKFold(10), scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            y_pred = grid.predict(X_test)
            y_pred_proba = grid.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            y_pred_adjusted = (y_pred_proba >= optimal_threshold).astype(int)
            acc_adjusted = accuracy_score(y_test, y_pred_adjusted)
            
            results[name] = {
                'best_params': grid.best_params_,
                'accuracy': acc,
                'adjusted_accuracy': acc_adjusted,
                'roc_auc': auc,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'feature_importances': dict(zip(grid.best_estimator_.named_steps['preprocessor'].get_feature_names_out(),
                                               grid.best_estimator_.named_steps['classifier'].feature_importances_)) if hasattr(grid.best_estimator_.named_steps['classifier'], 'feature_importances_') else None
            }
            
            if auc > best_auc:
                best_auc = auc
                best_model = grid.best_estimator_
                best_params[name] = {k.replace('classifier__', ''): v for k, v in grid.best_params_.items()}

        cv_results = {}
        for name, config in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', config['model'].set_params(**best_params.get(name, {})))
            ])
            cv_scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(10), scoring='roc_auc')
            cv_results[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}

        return results, cv_results, best_model
    except Exception as e:
        st.error(f"Ошибка при обучении моделей: {str(e)}")
        return None, None, None

# Load data
df, categorical_cols, numeric_cols = load_data()
if df is None:
    st.stop()

# Sidebar for navigation
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите раздел:", ["Обзор данных", "Матрица корреляций", "Результаты моделей", "Кросс-валидация", "Лучшая модель"])

# Data Overview
if page == "Обзор данных":
    st.subheader("Обзор данных")
    st.write("Первые 10 строк датасета:")
    st.dataframe(df.head(10))
    st.write("Статистика по данным:")
    st.dataframe(df.describe())

# Correlation Matrix
elif page == "Матрица корреляций":
    st.subheader("Матрица корреляций")
    correlation_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', linewidths=0.5)
    plt.title('Матрица корреляций', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Model Results
elif page == "Результаты моделей":
    st.subheader("Результаты моделей")
    results, cv_results, best_model = train_models(df, categorical_cols, numeric_cols)
    if results is None:
        st.stop()

    for name, result in results.items():
        st.write(f"**{name}**")
        st.write(f"Лучшие параметры: {result['best_params']}")
        st.write(f"Точность: {result['accuracy']:.4f}")
        st.write(f"Точность (с оптимальным порогом): {result['adjusted_accuracy']:.4f}")
        st.write(f"ROC AUC: {result['roc_auc']:.4f}")
        st.write("Отчет по классификации:")
        st.json(result['classification_report'])
        
        if result['feature_importances']:
            st.write("Важность признаков:")
            feature_df = pd.DataFrame(list(result['feature_importances'].items()), columns=['Признак', 'Важность']).sort_values('Важность', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Важность', y='Признак', data=feature_df.head(10), palette='viridis')
            plt.title(f'Топ-10 признаков по важности ({name})')
            st.pyplot(fig)

# Cross-Validation
elif page == "Кросс-валидация":
    st.subheader("Кросс-валидация")
    results, cv_results, best_model = train_models(df, categorical_cols, numeric_cols)
    if cv_results is None:
        st.stop()

    for name, cv in cv_results.items():
        st.write(f"{name} CV ROC AUC: {cv['mean']:.4f} (+/- {cv['std']:.4f})")

# Best Model
elif page == "Лучшая модель":
    st.subheader("Лучшая модель")
    results, cv_results, best_model = train_models(df, categorical_cols, numeric_cols)
    if best_model is None:
        st.stop()

    cv_scores_best = cross_val_score(best_model, df.drop('success', axis=1), df['success'], cv=StratifiedKFold(10), scoring='roc_auc')
    st.write(f"Лучшая модель CV ROC AUC: {cv_scores_best.mean():.4f} (+/- {cv_scores_best.std():.4f})")

    # Plot ROC AUC distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(cv_scores_best, kde=True, color='blue')
    plt.title('Распределение ROC AUC для лучшей модели (10-fold CV)')
    plt.xlabel('ROC AUC')
    plt.ylabel('Частота')
    st.pyplot(fig)
```
