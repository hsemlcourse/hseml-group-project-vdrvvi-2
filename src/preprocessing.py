# -*- coding: utf-8 -*-
"""
Метеопрогноз для Сегеда: предсказание температуры
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 1. Загрузка
print("1. Загрузка данных...")
df = pd.read_csv('data/raw/weatherHistory.csv')

print(f"Исходный объём: {df.shape}")

# 2. Очистка
print("\n2. Очистка данных...")

# Удаляем Apparent Temperature (ВАЖНО: избегаем дата-лика!)
if 'Apparent Temperature (C)' in df.columns:
    df = df.drop(columns=['Apparent Temperature (C)'])
    print("Удалён 'Apparent Temperature (C)' - основной источник дата-лика")

# Пропуски
df = df.dropna()

# Дубликаты
df = df.drop_duplicates()

# Выбросы по температуре
df = df[(df['Temperature (C)'] > -30) & (df['Temperature (C)'] < 45)]

# Удаляем Loud Cover
if 'Loud Cover' in df.columns and df['Loud Cover'].nunique() == 1:
    df = df.drop(columns=['Loud Cover'])

# Типы данных
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df = df.sort_values('Formatted Date').reset_index(drop=True)

print(f"После очистки: {df.shape}")

# 3. Feature engineering (базовый)
print("\n3. Feature engineering...")

# Временные признаки
df['Hour'] = df['Formatted Date'].dt.hour
df['DayOfYear'] = df['Formatted Date'].dt.dayofyear
df['Month'] = df['Formatted Date'].dt.month
df['DayOfWeek'] = df['Formatted Date'].dt.dayofweek
df['Year'] = df['Formatted Date'].dt.year

# Циклическое кодирование
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Категориальные признаки
categorical_cols = ['Summary', 'Precip Type', 'Daily Summary']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Удаляем временные столбцы
df = df.drop(columns=['Formatted Date'])

print(f"После FE: {df.shape[1]} признаков (включая таргет)")

# 4. Разделение данных
print("\n4. Разделение данных...")

X = df.drop(columns=['Temperature (C)'])
y = df['Temperature (C)']

train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size].copy()
y_train = y[:train_size].copy()
X_val = X[train_size:train_size+val_size].copy()
y_val = y[train_size:train_size+val_size].copy()
X_test = X[train_size+val_size:].copy()
y_test = y[train_size+val_size:].copy()

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 5. Добавление лагов (БЕЗ дата-лика)
print("\n5. Добавление лаговых признаков...")

def add_lag_features(X, y, lags=[1, 3, 6]):
    df_temp = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df_temp.columns = list(X.columns) + ['Temperature (C)']
    
    for lag in lags:
        df_temp[f'Temp_lag_{lag}h'] = df_temp['Temperature (C)'].shift(lag)
    
    df_temp = df_temp.dropna().reset_index(drop=True)
    X_new = df_temp.drop(columns=['Temperature (C)'])
    y_new = df_temp['Temperature (C)']
    return X_new, y_new

X_train, y_train = add_lag_features(X_train, y_train)
X_val, y_val = add_lag_features(X_val, y_val)
X_test, y_test = add_lag_features(X_test, y_test)

print(f"После лагов: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Количество признаков: {X_train.shape[1]}")

# 6. Сохранение
os.makedirs('data/processed', exist_ok=True)
df_final = pd.concat([X_train, y_train], axis=1)
df_final = pd.concat([df_final, pd.concat([X_val, y_val], axis=1)])
df_final = pd.concat([df_final, pd.concat([X_test, y_test], axis=1)])
df_final.to_csv('data/processed/processed_weather.csv', index=False)

print("\n------------------------ Данные сохранены: data/processed/processed_weather.csv")

# 7. Baseline на исправленных данных
print("\n" + "=" * 60)
print("BASELINE НА ИСПРАВЛЕННЫХ ДАННЫХ")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor

# Простая модель для проверки
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

print(f"Random Forest (50 деревьев) на исправленных данных:")
print(f"MAE: {mean_absolute_error(y_val, y_pred):.2f}°C")
print(f"R²: {r2_score(y_val, y_pred):.4f}")
