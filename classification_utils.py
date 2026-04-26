"""
Модуль для обучения и сравнения моделей КЛАССИФИКАЦИИ

Предоставляет функции для запуска нескольких моделей с подбором
гиперпараметров через GridSearchCV и сравнением результатов.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')


# Возвращает словарь моделей для классификации и сетки гиперпараметров
def get_classification_models_with_params():

    models = {
        # Логистическая регрессия
        'LogisticRegression': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state=42, max_iter=5000))
            ]),
            {
                'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'model__penalty': ['l2']
            }
        ),

        # Decision Tree
        'DecisionTree': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', DecisionTreeClassifier(random_state=42))
            ]),
            {
                'model__max_depth': [5, 10, None],
                'model__criterion': ['gini', 'entropy']
            }
        ),

        # Random Forest
        'RandomForest': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=42, n_jobs=1))
            ]),
            {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, None]
            }
        ),

        # Gradient Boosting
        'GradientBoosting': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(random_state=42))
            ]),
            {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.1]
            }
        ),

        # SVM
        'SVC': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(random_state=42, probability=True))
            ]),
            {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['linear', 'rbf']
            }
        ),

        # K-Nearest Neighbors
        'KNN': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier(n_jobs=1))
            ]),
            {
                'model__n_neighbors': [3, 5, 7, 10, 15],
                'model__weights': ['uniform', 'distance']
            }
        )
    }
    return models


# Считает метрики качества для бинарной классификации
def evaluate_classification(y_true, y_pred, y_proba=None, model_name=''):

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_proba is not None:
        try:
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics['ROC_AUC'] = np.nan

    return metrics


# Обучает несколько моделей классификации, подбирает гиперпараметры
# и возвращает таблицу сравнения
def train_and_compare_classification(X, y, target_name='target', cv=5,
                                     test_size=0.3, random_state=42):

    # Разделение на train и test со стратификацией
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models_dict = get_classification_models_with_params()

    results = []
    best_models = {}

    for model_name, (pipeline, param_grid) in models_dict.items():

        if not param_grid:
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            best_params = {}
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                        cv=cv, scoring='f1', n_jobs=-1)
            best_cv_score = cv_scores.mean()
        else:
            # GridSearchCV с метрикой F1
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            best_params = grid.best_params_
            best_cv_score = grid.best_score_

        # Предсказания
        y_pred_train = best_estimator.predict(X_train)
        y_pred_test = best_estimator.predict(X_test)

        # Вероятности для ROC AUC
        try:
            y_proba_test = best_estimator.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba_test = None

        train_metrics = evaluate_classification(y_train, y_pred_train)
        test_metrics = evaluate_classification(y_test, y_pred_test, y_proba_test)

        results.append({
            'Model': model_name,
            'Best params': str(best_params),
            'CV F1': best_cv_score,
            'Train Accuracy': train_metrics['Accuracy'],
            'Test Accuracy': test_metrics['Accuracy'],
            'Test Precision': test_metrics['Precision'],
            'Test Recall': test_metrics['Recall'],
            'Test F1': test_metrics['F1'],
            'Test ROC_AUC': test_metrics.get('ROC_AUC', np.nan)
        })

        best_models[model_name] = best_estimator

    results_df = pd.DataFrame(results)
    # Сортируем по F1
    results_df = results_df.sort_values('Test F1', ascending=False).reset_index(drop=True)

    return results_df, best_models, (X_train, X_test, y_train, y_test)


# Итоговая таблица сравнения моделей
def print_results_table(results_df, target_name=''):

    cols_to_show = ['Model', 'CV F1', 'Test Accuracy', 'Test Precision',
                    'Test Recall', 'Test F1', 'Test ROC_AUC']
    print(results_df[cols_to_show].to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    best_test_f1 = results_df.iloc[0]['Test F1']
    print(f"\nЛучшая модель: {best_model_name} (Test F1 = {best_test_f1:.4f})")


# Матрица ошибок
def print_confusion_matrix(y_true, y_pred, model_name=''):

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN (правильно отрицательные): {tn}")
    print(f"FP (ложные срабатывания):     {fp}")
    print(f"FN (пропущенные положительные): {fn}")
    print(f"TP (правильно положительные): {tp}")
    print(classification_report(y_true, y_pred, zero_division=0))
