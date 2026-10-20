import pandas as pd
import numpy as np
import time

np.random.seed(42)

# ---- Load data ----
ratings = pd.read_csv('data/ratings.dat', sep='::', engine='python',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Re-number users and movies as 0,1,2,...
ratings['u'] = ratings['user_id'].astype('category').cat.codes
ratings['i'] = ratings['movie_id'].astype('category').cat.codes
n_users = ratings['u'].nunique()
n_items = ratings['i'].nunique()

# ---- Train/test split (80/20) ----
shuffled = ratings.sample(frac=1, random_state=42).reset_index(drop=True)
cut = int(0.8 * len(shuffled))
train, test = shuffled[:cut], shuffled[cut:]

def rmse(y, p): return np.sqrt(np.mean((y - p) ** 2))
def mae(y, p): return np.mean(np.abs(y - p))

def precision_recall_at_k(test_df, preds, k=10, thr=4.0):
    d = test_df[['u', 'rating']].copy()
    d['pred'] = preds
    precs, recs = [], []
    for _, g in d.groupby('u'):
        g = g.sort_values('pred', ascending=False)
        n_rel = (g['rating'] >= thr).sum()
        top = g.head(k)
        n_rec = (top['pred'] >= thr).sum()
        n_hit = ((top['pred'] >= thr) & (top['rating'] >= thr)).sum()
        precs.append(n_hit / n_rec if n_rec else 0)
        recs.append(n_hit / n_rel if n_rel else 0)
    return np.mean(precs), np.mean(recs)

results = []
def evaluate(name, preds):
    preds = np.clip(preds, 1, 5)
    p, r = precision_recall_at_k(test, preds)
    results.append({'Model': name,
                    'RMSE': rmse(test['rating'].values, preds),
                    'MAE': mae(test['rating'].values, preds),
                    'Precision@10': p, 'Recall@10': r})
    print(f"{name}: RMSE={results[-1]['RMSE']:.4f}")

global_mean = train['rating'].mean()

# ---- Model 1: Item mean baseline ----
item_mean = train.groupby('i')['rating'].mean()
p1 = test['i'].map(item_mean).fillna(global_mean).values
evaluate('Item Mean Baseline', p1)

# ---- Model 2: Bias baseline ----
item_bias = (train.groupby('i')['rating'].mean() - global_mean)
tmp = train.copy()
tmp['ib'] = tmp['i'].map(item_bias)
user_bias = (tmp['rating'] - global_mean - tmp['ib']).groupby(tmp['u']).mean()
p2 = (global_mean + test['u'].map(user_bias).fillna(0).values
      + test['i'].map(item_bias).fillna(0).values)
evaluate('Bias Baseline', p2)

# ---- Model 3: SVD (matrix factorization, SGD) ----
K, EPOCHS, LR, REG = 50, 15, 0.01, 0.05
U = np.random.normal(0, 0.1, (n_users, K))
V = np.random.normal(0, 0.1, (n_items, K))
bu = np.zeros(n_users)
bi = np.zeros(n_items)

tu, ti, tr = train['u'].values, train['i'].values, train['rating'].values
start = time.time()
for epoch in range(EPOCHS):
    order = np.random.permutation(len(tr))
    for idx in order:
        u, i, r = tu[idx], ti[idx], tr[idx]
        pred = global_mean + bu[u] + bi[i] + U[u] @ V[i]
        e = r - pred
        bu[u] += LR * (e - REG * bu[u])
        bi[i] += LR * (e - REG * bi[i])
        Uu = U[u].copy()
        U[u] += LR * (e * V[i] - REG * U[u])
        V[i] += LR * (e * Uu - REG * V[i])
    pt = global_mean + bu[test['u'].values] + bi[test['i'].values] \
         + np.sum(U[test['u'].values] * V[test['i'].values], axis=1)
    print(f"  epoch {epoch+1}/{EPOCHS}  test RMSE={rmse(test['rating'].values, np.clip(pt,1,5)):.4f}"
          f"  ({time.time()-start:.0f}s)")

evaluate('SVD (own implementation)', pt)

# ---- Save results and model ----
table = pd.DataFrame(results).round(4)
print("\n", table.to_string(index=False))
table.to_csv('results/metrics.csv', index=False)
np.savez('results/svd_model.npz', U=U, V=V, bu=bu, bi=bi, gm=global_mean)
print("\nSaved results/metrics.csv")