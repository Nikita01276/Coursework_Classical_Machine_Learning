"""
Функции предобработки данных
Содержит функции, которые используются во всех 7 анализах
 1. Удаление служебного столбца Unnamed: 0
 2. Удаление константных и почти константных признаков 
 3. Заполнение пропусков медианой соответствующего столбца
 4. Удаление одного признака из каждой пары с корреляцией > 0.95
"""

import numpy as np
import pandas as pd


# Список целевых столбцов, который используется во всех файлах
TARGET_COLUMNS = ['IC50, mM', 'CC50, mM', 'SI']
# Загружает датасет и удаляет служебный столбец-индекс
def load_raw_data(path='data.xlsx'):

    df = pd.read_excel(path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df

# Удаляет константные и почти константные признаки из датасета
def remove_constant_features(df, target_columns, threshold=0.99):

    # Получаем список признаков(без целевых)
    feature_cols = [col for col in df.columns if col not in target_columns]

    # Ищем признаки для удаления
    to_drop = []
    for col in feature_cols:
        n_unique = df[col].nunique()
        # Константный признак 
        if n_unique <= 1:
            to_drop.append(col)
        else:
            #Доля самого частого значения
            top_freq = df[col].value_counts(normalize=True).iloc[0]
            if top_freq > threshold:
                to_drop.append(col)

    # Удаляем признаки
    df_clean = df.drop(columns=to_drop)
    return df_clean, to_drop

# Заполняем пропуски медианой соответствующего столбца
def fill_missing_with_median(df, target_columns):

    df_filled = df.copy()
    feature_cols = [col for col in df.columns if col not in target_columns]

    # Заполняем пропуски в каждом признаке его медианой
    for col in feature_cols:
        if df_filled[col].isnull().sum() > 0:
            median_value = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_value)

    return df_filled

#Удаляет один признак из каждой пары с высокой корреляцией
def remove_highly_correlated_features(df, target_columns, threshold=0.95):

    feature_cols = [col for col in df.columns if col not in target_columns]

    # Считаем корреляционную матрицу для признаков
    corr_matrix = df[feature_cols].corr().abs()

    # Берем верхний треугольник матрицы (без диагонали), чтобы каждая пара рассматривалась только один раз
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Признаки, у которых хотя бы с одним другим корреляция > threshold, будем удалять 
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

# функция предобработки данных
def preprocess_data(path='data.xlsx', verbose=True):

    # загрузка
    df = load_raw_data(path)
    if verbose:
        print(f"Загружено: {df.shape[0]} объектов, {df.shape[1]} столбцов")

    # удаление константных признаков
    df, dropped_const = remove_constant_features(df, TARGET_COLUMNS)
    if verbose:
        print(f"Удалено константных признаков: {len(dropped_const)}")

    # заполнение пропусков медианой
    df = fill_missing_with_median(df, TARGET_COLUMNS)
    if verbose:
        print("Пропуски заполнены медианой каждого столбца")

    # удаление скоррелированных признаков
    df, dropped_corr = remove_highly_correlated_features(df, TARGET_COLUMNS,
                                                          threshold=0.95)
    if verbose:
        print(f"Удалено признаков с корреляцией > 0.95: {len(dropped_corr)}")
        print(f"Итого: {df.shape[0]} объектов, {df.shape[1]} столбцов "
              f"({df.shape[1] - len(TARGET_COLUMNS)} признаков)")

    return df

# Извлекаем признаки X и целевую переменную y из датафрейма
def get_features_and_target(df, target_name, task_type='regression',
                             threshold=None):

    # Удаляем все целевые столбцы из X
    X = df.drop(columns=TARGET_COLUMNS)
    y_raw = df[target_name]

    if task_type == 'regression':
        # Применяем логарифмирование так как распределение сильно скошенное
        y = np.log1p(y_raw)
    elif task_type == 'classification':
        # Для классификации
        if threshold is None:
            # По умолчанию медиана
            threshold = y_raw.median()
        # 1 если значение выше порога, иначе 0
        y = (y_raw > threshold).astype(int)
    else:
        raise ValueError(f"Неизвестный тип задачи: {task_type}")

    return X, y
