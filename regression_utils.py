"""
Модуль для обучения и сравнения моделей регрессии

Функции для запуска нескольких моделей с подбором гиперпараметров через GridSearchCV и сравнением результатов
"""
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings('ignore')


# Возвращает словарь моделей для регрессии и сетки гиперпараметров
def get_regression_models_with_params():
    models = {
        # Линейная регрессия (без гиперпараметров)
        'LinearRegression': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            {} 
        ),

        # Kинейная регрессия
        'Ridge': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(random_state=42))
            ]),
            {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        ),

        # Lasso 
        'Lasso': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(random_state=42, max_iter=10000))
            ]),
            {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            }
        ),

        # Decision Tree
        'DecisionTree': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', DecisionTreeRegressor(random_state=42))
            ]),
            {
                'model__max_depth': [5, 10, None],
                'model__min_samples_leaf': [1, 5]
            }
        ),

        # Random Forest 
        # Используем уменьшенную сетку для ускорения
        'RandomForest': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(random_state=42, n_jobs=1))
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
                ('model', GradientBoostingRegressor(random_state=42))
            ]),
            {
                'model__n_estimators': [100, 200],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.1]
            }
        ),

        # Support Vector Regression
        'SVR': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR())
            ]),
            {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['linear', 'rbf']
            }
        ),

        #K-Nearest Neighbors
        'KNN': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsRegressor(n_jobs=1))
            ]),
            {
                'model__n_neighbors': [3, 5, 7, 10, 15],
                'model__weights': ['uniform', 'distance']
            }
        )
    }
    return models

#Считает метрики качества для задачи регрессии
def evaluate_regression(y_true, y_pred, model_name=''):

    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

# обучает несколько моделей с подбором гиперпараметрови возвращает таблицу сравнения
def train_and_compare_regression(X, y, target_name='target', cv=5,
                                  test_size=0.3, random_state=42):

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    #Получаем словарь моделей
    models_dict = get_regression_models_with_params()

    results = []  # Список словарей с результатами
    best_models = {}  # Лучшие обученные модели

    # Перебираем все модели
    for model_name, (pipeline, param_grid) in models_dict.items():

        # если у модели нет гиперпараметров, тогда просто обучаем
        if not param_grid:
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            best_params = {}
            # Считаем кросс-валидацию вручную
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                         cv=cv, scoring='r2', n_jobs=-1)
            best_cv_score = cv_scores.mean()
        else:
            # Используем GridSearchCV для подбора гиперпараметров
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            best_params = grid.best_params_
            best_cv_score = grid.best_score_

        # Считаем метрики на тестовой выборке
        y_pred_train = best_estimator.predict(X_train)
        y_pred_test = best_estimator.predict(X_test)

        train_metrics = evaluate_regression(y_train, y_pred_train)
        test_metrics = evaluate_regression(y_test, y_pred_test)

        # Сохраняем результаты в строку таблицы
        results.append({
            'Model': model_name,
            'Best params': str(best_params),
            'CV R2': best_cv_score,
            'Train R2': train_metrics['R2'],
            'Test R2': test_metrics['R2'],
            'Test MAE': test_metrics['MAE'],
            'Test RMSE': test_metrics['RMSE']
        })

        best_models[model_name] = best_estimator


    # Итоговая таблица
    results_df = pd.DataFrame(results)
    #Сортируем по Test R2 
    results_df = results_df.sort_values('Test R2', ascending=False).reset_index(drop=True)

    return results_df, best_models, (X_train, X_test, y_train, y_test)

# Итоговая таблица сравнения моделей
def print_results_table(results_df, target_name=''):

    print(f"Итоговая таблица сравнений для {target_name}")

    #Печатаем только основные столбцы
    cols_to_show = ['Model', 'CV R2', 'Train R2', 'Test R2', 'Test MAE', 'Test RMSE']
    print(results_df[cols_to_show].to_string(index=False))

    # Лучшая модель
    best_model_name = results_df.iloc[0]['Model']
    best_test_r2 = results_df.iloc[0]['Test R2']
    print(f"\n Лучшая модель: {best_model_name} (Test R2 = {best_test_r2:.4f}) ***")
