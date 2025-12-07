"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from scipy import sparse
import os

SEED = 993
np.random.seed(SEED)

def calculate_dcg(scores, relevances, k=10):
    """DCG@k - Discounted Cumulative Gain"""
    order = np.argsort(-scores)
    top_k = order[:k]
    rel_top_k = relevances[top_k]
    
    gains = 2**rel_top_k - 1
    discounts = 1.0 / np.log2(np.arange(2, len(rel_top_k) + 2))
    return np.sum(gains * discounts)

def calculate_ndcg_at_k(relevances, scores, k=10):
    """nDCG@k - Normalized Discounted Cumulative Gain"""
    dcg = calculate_dcg(scores, relevances, k)
    
    ideal_order = np.argsort(-relevances)
    ideal_top_k = ideal_order[:k]
    ideal_rel_top_k = relevances[ideal_top_k]
    
    ideal_gains = 2**ideal_rel_top_k - 1
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_rel_top_k) + 2))
    idcg = np.sum(ideal_gains * ideal_discounts)
    
    return dcg / idcg if idcg > 0 else 0.0

def build_features(df, tfidf_vectorizer=None, top_categories_dict=None, is_train=True):
    df = df.copy()
    
    #Заполнение пропусков
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
    
    #Создание текстовых фичей
    df['product_text'] = (
        df['product_title'].astype(str) + ' ' +
        df['product_description'].astype(str) + ' ' +
        df['product_bullet_point'].astype(str)
    )
    
    #TF-IDF фичи
    if is_train:
        tfidf = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf.fit(df['product_text'])
        tfidf_vectorizer = tfidf
    else:
        tfidf = tfidf_vectorizer
    
    query_tfidf = tfidf.transform(df['query'])
    product_tfidf = tfidf.transform(df['product_text'])
    
    #Вычисляем косинусное сходство
    query_norm = np.sqrt(np.asarray(query_tfidf.power(2).sum(axis=1)).flatten())
    product_norm = np.sqrt(np.asarray(product_tfidf.power(2).sum(axis=1)).flatten())
    
    dot_product = np.asarray((query_tfidf.multiply(product_tfidf)).sum(axis=1)).flatten()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = dot_product / (query_norm * product_norm)
        similarity = np.nan_to_num(similarity)
    
    df['tfidf_sim'] = similarity
    
    #Лексические фичи
    def lexical_features(row):
        query = str(row['query']).lower()
        title = str(row['product_title']).lower()
        desc = str(row['product_description']).lower()
        
        query_words = set(query.split())
        title_words = set(title.split())
        desc_words = set(desc.split())
        
        features = {
            'jaccard_title': len(query_words & title_words) / max(len(query_words | title_words), 1),
            'jaccard_desc': len(query_words & desc_words) / max(len(query_words | desc_words), 1),
            
            'overlap_title': len(query_words & title_words) / max(len(query_words), 1),
            'overlap_desc': len(query_words & desc_words) / max(len(query_words), 1),
            
            'query_len_chars': len(query),
            'title_len_chars': len(title),
            'desc_len_chars': len(desc),
            'query_len_words': len(query_words),
            'title_len_words': len(title_words),
            'desc_len_words': len(desc_words),
        }
        
        features['exact_match'] = 1 if query in title or query in desc else 0
        
        if 'product_brand' in row and row['product_brand']:
            brand = str(row['product_brand']).lower()
            features['brand_in_query'] = 1 if brand in query else 0
            features['brand_in_title'] = 1 if brand in title else 0
        
        return pd.Series(features)
    
    lexical_feats = df.apply(lexical_features, axis=1)
    for col in lexical_feats.columns:
        df[col] = lexical_feats[col]
    
    #Категориальные фичи
    if is_train:
        top_categories_dict = {}
        categorical_cols = ['product_locale', 'product_color'] if 'product_locale' in df.columns else []
        for cat_col in categorical_cols:
            if cat_col in df.columns:
                #Сохраняем топ-10 категорий из тренировочных данных
                top_cats = df[cat_col].value_counts().head(10).index.tolist()
                top_categories_dict[cat_col] = top_cats
                
                #Создаем one-hot признаки для тренировочных данных
                for cat in top_cats:
                    df[f'{cat_col}_{cat}'] = (df[cat_col] == cat).astype(int)
    else:
        #Для тестовых данных используем сохраненные категории из тренировочных
        if top_categories_dict:
            for cat_col, top_cats in top_categories_dict.items():
                if cat_col in df.columns:
                    for cat in top_cats:
                        df[f'{cat_col}_{cat}'] = (df[cat_col] == cat).astype(int)
    
    df = df.drop(columns=['product_text'], errors='ignore')
    
    return df, tfidf_vectorizer, top_categories_dict

def train_with_cv(df, params, n_folds=5):
    
    exclude_cols = ['id', 'query_id', 'query', 'product_title', 
                    'product_description', 'product_bullet_point',
                    'product_text', 'relevance', 'product_brand', 
                    'product_color', 'product_locale']
    
    numeric_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            try:
                sample = df[col].dropna().iloc[:10]
                pd.to_numeric(sample, errors='raise')
                numeric_cols.append(col)
            except:
                print(f"Пропускаем нечисловой признак: {col}")
    
    print(f"Используем {len(numeric_cols)} числовых признаков")
    
    X = df[numeric_cols].values
    y = df['relevance'].values
    groups = df['query_id'].values
    
    gkf = GroupKFold(n_splits=n_folds)
    fold_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_query_ids = df.iloc[train_idx]['query_id']
        val_query_ids = df.iloc[val_idx]['query_id']
        
        group_train = train_query_ids.value_counts().sort_index().values
        group_val = val_query_ids.value_counts().sort_index().values
        
        lgb_train = lgb.Dataset(X_train, y_train, group=group_train)
        lgb_val = lgb.Dataset(X_val, y_val, group=group_val, reference=lgb_train)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,  
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        
        y_pred = model.predict(X_val)
        
        val_df = df.iloc[val_idx].copy()
        val_df['prediction'] = y_pred
        
        ndcg_scores = []
        
        for query_id in val_df['query_id'].unique():
            q_data = val_df[val_df['query_id'] == query_id]
            ndcg = calculate_ndcg_at_k(
                q_data['relevance'].values, 
                q_data['prediction'].values, 
                k=10
            )
            ndcg_scores.append(ndcg)
        
        fold_ndcg = np.mean(ndcg_scores)
        
        print(f"Fold {fold+1} - nDCG@10: {fold_ndcg:.4f}")
        fold_scores.append(fold_ndcg)
        models.append(model)
    
    print(f"\n{'='*50}")
    print(f"CV Results:")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: nDCG@10 = {score:.4f}")
    print(f"Mean CV nDCG@10: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    return models, np.mean(fold_scores), numeric_cols  

def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    import os
    import pandas as pd
    
    test = pd.read_csv('test.csv')
    
    submission = pd.DataFrame({
        'id': test['id'].values,
        'prediction': predictions,
    })
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    df_train_full, tfidf_vec, top_categories_dict = build_features(train, is_train=True)
    test_features, _, _ = build_features(test, tfidf_vectorizer=tfidf_vec, 
                                         top_categories_dict=top_categories_dict, is_train=False)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'seed': SEED,
        'label_gain': [0, 1, 3, 7],  # 2^0-1=0, 2^1-1=1, 2^2-1=3, 2^3-1=7
    }
    

    cv_models, cv_score, feature_cols = train_with_cv(df_train_full, params, n_folds=3)
    
    X_train = df_train_full[feature_cols].values
    y_train = df_train_full['relevance'].values
    
    group_sizes = df_train_full.groupby('query_id').size().values
    
    lgb_train_full = lgb.Dataset(X_train, y_train, group=group_sizes)
    
    if cv_models:
        best_iteration = int(np.mean([model.best_iteration for model in cv_models]))
        print(f"Среднее лучшее количество итераций по фолдам: {best_iteration}")
    else:
        best_iteration = 300
        print(f"Используем фиксированное количество итераций: {best_iteration}")
    
    final_model = lgb.train(
        params,
        lgb_train_full,
        num_boost_round=best_iteration,
        valid_sets=[lgb_train_full],
        valid_names=['train'],
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )
    
    missing_features = [f for f in feature_cols if f not in test_features.columns]
    if missing_features:
        print(f"Добавляем отсутствующие признаки в тест: {missing_features}")
        for f in missing_features:
            test_features[f] = 0  
    
    X_test = test_features[feature_cols].values
    predictions = final_model.predict(X_test)
    
    create_submission(predictions)
        
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()