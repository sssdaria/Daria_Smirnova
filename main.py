import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import os

SEED = 42
np.random.seed(SEED)

def load_and_preprocess_data(train_path='train.csv', test_path='test.csv'):
    """Загрузка и базовая предобработка данных"""
    print('='*60)
    print('1. Загрузка данных')
    print('='*60)
    
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except:
        # Fallback на синтетические данные как в оригинальном main.py
        print("CSV не найдены, генерируем синтетические данные")
        train = pd.DataFrame({
            'product_id': np.arange(100),
            'dt': pd.date_range('2023-01-01', periods=100, freq='D'),
            'price_p05': np.random.uniform(10, 50, 100),
            'price_p95': np.random.uniform(60, 120, 100),
            'nstores': np.random.randint(1, 10, 100),
            'activity_flag': np.random.randint(0, 2, 100)
        })
        test = pd.DataFrame({
            'product_id': np.arange(100, 150),
            'dt': pd.date_range('2023-04-10', periods=50, freq='D'),
            'nstores': np.random.randint(1, 10, 50),
            'activity_flag': np.random.randint(0, 2, 50)
        })
    
    train['is_train'] = 1
    test['is_train'] = 0
    
    train = train.sort_values(['product_id', 'dt'])
    test = test.sort_values(['product_id', 'dt'])
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Временные фичи
    df['dt'] = pd.to_datetime(df['dt'])
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    df['week'] = df['dt'].dt.isocalendar().week
    df['quarter'] = df['dt'].dt.quarter
    df['dayofyear'] = df['dt'].dt.dayofyear
    df['season'] = df['month'].isin([12,1,2]).astype(int)
    df['is_monthend'] = df['dt'].dt.is_month_end.astype(int)
    df['is_monthstart'] = df['dt'].dt.is_month_start.astype(int)
    
    # Ценовые фичи
    if 'price_p05' in df.columns and 'price_p95' in df.columns:
        df['price_width'] = df['price_p95'] - df['price_p05']
        df['price_mid'] = (df['price_p05'] + df['price_p95']) / 2
    
    print(f"train_len={len(train)}, test_len={len(test)}, total_len={len(df)}")
    return df, train, test

def create_time_features_safe(df, target_cols, train_mask):
    """Создание лагов и скользящих окон"""
    result = df.copy()
    result = result.sort_values(['product_id', 'dt'])
    
    for col in target_cols:
        if col not in result.columns:
            continue
        grouper = result.groupby(['product_id'])[col]
        
        for lag in [1, 3, 7, 14]:
            result[f'{col}_lag{lag}'] = grouper.shift(lag)
        
        for window in [3, 7, 14]:
            shifted = grouper.shift(1)
            result[f'{col}_roll_mean{window}'] = shifted.rolling(window, min_periods=1).mean()
            result[f'{col}_roll_std{window}'] = shifted.rolling(window, min_periods=1).std()
    
    return result

def create_product_embeddings(df_train):
    """Упрощенные эмбеддинги продуктов"""
    product_ids = df_train['product_id'].unique()
    embeddings = pd.DataFrame({'product_id': product_ids})
    
    if 'price_p05' in df_train.columns:
        price_stats = df_train.groupby('product_id')['price_p05'].agg(['mean', 'std']).reset_index()
        embeddings = embeddings.merge(price_stats, on='product_id', how='left').fillna(0)
        embeddings['product_cluster'] = pd.qcut(embeddings['mean'], 10, labels=False, duplicates='drop')
    
    return embeddings

def detect_anomalies(df):
    """Обнаружение аномалий"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['price_p05', 'price_p95', 'is_train', 'product_id', 'year']
    
    feature_cols = [col for col in numeric_cols 
                   if col not in exclude and df[col].nunique() > 1][:10]
    
    if len(feature_cols) < 3:
        df['is_anomaly'] = 0
        return df
    
    X = df[feature_cols].fillna(0)
    iso = IsolationForest(random_state=SEED, contamination=0.1)
    df['is_anomaly'] = iso.fit_predict(X) == -1
    return df

def create_uplift_features(df):
    """Uplift фичи"""
    result = df.copy()
    if 'activity_flag' in result.columns:
        result['prev_activity'] = result.groupby('product_id')['activity_flag'].shift(1).fillna(0)
        result['activity_count_7d'] = result.groupby('product_id')['activity_flag'].transform(
            lambda x: x.rolling(7, min_periods=1).sum()
        )
    return result

def encode_categorical_features(train_df, val_df, test_df, cat_cols):
    """Label encoding"""
    encoders = {}
    for col in cat_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str).fillna('missing'))
            encoders[col] = le
    
    def encode_df(df, encoders):
        df_enc = df.copy()
        for col, le in encoders.items():
            if col in df_enc.columns:
                df_enc[col] = df_enc[col].astype(str).fillna('missing').map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df_enc
    
    val_df = encode_df(val_df, encoders)
    test_df = encode_df(test_df, encoders)
    return train_df, val_df, test_df, encoders

def create_ensemble_model_lower():
    """Ensemble для нижней границы"""
    return CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=7,
        loss_function='Quantile:alpha=0.05', random_seed=SEED, verbose=0
    )

def create_ensemble_model_upper():
    """Ensemble для верхней границы"""
    return CatBoostRegressor(
        iterations=800, learning_rate=0.05, depth=7,
        loss_function='Quantile:alpha=0.95', random_seed=SEED, verbose=0
    )

def postprocess_predictions(pred_min, pred_max, min_width=0.1):
    """Постобработка предсказаний"""
    pred_min = np.array(pred_min)
    pred_max = np.array(pred_max)
    
    pred_min_adj = np.minimum(pred_min, pred_max)
    pred_max_adj = np.maximum(pred_min, pred_max)
    
    width = pred_max_adj - pred_min_adj
    mask = width < min_width
    if mask.any():
        adjustment = (min_width - width[mask]) / 2
        pred_min_adj[mask] -= adjustment
        pred_max_adj[mask] += adjustment
    
    return pred_min_adj, pred_max_adj

def train_and_predict():
    """Основная функция обучения и предсказания (аналог main из notebook)"""
    print('='*60)
    print('НАЧИНАЕМ ОБУЧЕНИЕ')
    print('='*60)
    
    # 1. Загрузка
    df, train, test = load_and_preprocess_data()
    
    # 2. Временные фичи
    train_mask = df['is_train'] == 1
    if 'price_p05' in df.columns:
        # Сохраняем оригинальные цены для тренировки
        train_avg_p05 = df.loc[train_mask, 'price_p05'].mean()
        train_avg_p95 = df.loc[train_mask, 'price_p95'].mean()
        
        # Создаем временные фичи
        df = create_time_features_safe(df, ['price_p05', 'price_p95'], train_mask)
    
    # 3. Эмбеддинги
    train_df_for_emb = df[train_mask].copy()
    embeddings = create_product_embeddings(train_df_for_emb)
    df = df.merge(embeddings, on='product_id', how='left')
    
    # 4. Uplift
    df = create_uplift_features(df)
    
    # 5. Аномалии
    df = detect_anomalies(df)
    
    # 6. Подготовка фич
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    
    cat_cols = ['product_cluster']  # Упростим категориальные фичи
    cat_cols = [col for col in cat_cols if col in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('missing')
    
    train_df = df[df['is_train'] == 1].copy()
    test_df = df[df['is_train'] == 0].copy()
    
    exclude_cols = ['dt', 'price_p05', 'price_p95', 'is_train', 'price_width', 
                   'price_mid', 'product_id', 'year', 'week', 'quarter', 
                   'dayofyear', 'is_monthend', 'is_monthstart']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Feature cols: {len(feature_cols)}, Cat cols: {len(cat_cols)}")
    
    # 7. Сплит - УПРОЩЕННЫЙ ВАРИАНТ
    # Берем 80% для тренировки, 20% для валидации
    train_df = train_df.sort_values('dt')
    split_idx = int(len(train_df) * 0.8)
    train_part = train_df.iloc[:split_idx].copy()
    val_part = train_df.iloc[split_idx:].copy()
    
    X_train = train_part[feature_cols]
    X_val = val_part[feature_cols]
    y_train_min = train_part['price_p05']
    y_train_max = train_part['price_p95']
    y_val_min = val_part['price_p05']
    y_val_max = val_part['price_p95']
    X_test = test_df[feature_cols]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Проверка, что данные не пустые
    if len(X_train) == 0:
        print("ПРЕДУПРЕЖДЕНИЕ: X_train пустой! Используем все данные для тренировки")
        X_train = train_df[feature_cols]
        y_train_min = train_df['price_p05']
        y_train_max = train_df['price_p95']
        X_val = test_df[feature_cols]  # Валидация на тестовых данных
        y_val_min = pd.Series([0] * len(X_val))  # Заглушки
        y_val_max = pd.Series([0] * len(X_val))
    
    # 8. Encoding
    X_train, X_val, X_test, encoders = encode_categorical_features(
        X_train, X_val, X_test, cat_cols
    )
    
    # 9. Модели - УПРОЩЕННЫЙ ВАРИАНТ ДЛЯ ТЕСТИРОВАНИЯ
    print("Обучаем lower модель...")
    
    # Проверяем, есть ли данные
    if len(X_train) > 0 and len(y_train_min) > 0:
        model_lower = create_ensemble_model_lower()
        model_lower.fit(X_train, y_train_min)
        
        print("Обучаем upper модель...")
        model_upper = create_ensemble_model_upper()
        model_upper.fit(X_train, y_train_max)
        
        # 10. Предсказания тест
        test_pred_min_raw = model_lower.predict(X_test)
        test_pred_max_raw = model_upper.predict(X_test)
        test_pred_min, test_pred_max = postprocess_predictions(test_pred_min_raw, test_pred_max_raw)
    else:
        # Если данных нет, используем простые предсказания
        print("Нет тренировочных данных. Используем простые предсказания...")
        test_pred_min = np.ones(len(X_test)) * 30.0  # Примерные значения
        test_pred_max = np.ones(len(X_test)) * 80.0
    
    predictions = {
        'price_p05': test_pred_min,
        'price_p95': test_pred_max
    }
    
    return predictions, test_df

def create_submission(predictions, test_df=None):
    """Создание submission файла"""
    print('='*50)
    print('СОЗДАЕМ SUBMISSION')
    print('='*50)
    
    if test_df is not None:
        row_ids = (test_df['product_id'].astype(str) + '_' + test_df['dt'].astype(str))
    else:
        # Fallback
        n_predictions = len(predictions['price_p05'])
        row_ids = [f"row_{i}" for i in range(n_predictions)]
    
    submission = pd.DataFrame({
        'row_id': row_ids,
        'price_p05': predictions['price_p05'],
        'price_p95': predictions['price_p95']
    })
    
    submission['price_p05'] = submission['price_p05'].clip(lower=0)
    submission['price_p95'] = submission['price_p95'].clip(lower=submission['price_p05'] * 1.001)
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f'Submission saved: {submission_path}')
    print(f"Shape: {submission.shape}")
    print(f"Mean width: {(submission['price_p95'] - submission['price_p05']).mean():.2f}")
    print(f"p05 min: {submission['price_p05'].min():.2f}, max: {submission['price_p95'].max():.2f}")
    
    return submission_path

def main():
    """Главная функция"""
    print('='*50)
    print('НАЧАЛО!')
    print('='*50)
    
    predictions, test_df = train_and_predict()
    submission_path = create_submission(predictions, test_df)
    
    print('='*50)
    print('ВСЕ ГОТОВО!')
    print('='*50)
    print(f'Submission: {submission_path}')
    
    return submission_path

if __name__ == '__main__':
    main()
