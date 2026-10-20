import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---- Load data ----
ratings = pd.read_csv('data/ratings.dat', sep='::', engine='python',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('data/movies.dat', sep='::', engine='python',
                     names=['movie_id', 'title', 'genres'], encoding='latin-1')

print("Ratings:", ratings.shape)
print("Users:", ratings['user_id'].nunique())
print("Movies:", ratings['movie_id'].nunique())
print("Missing values:", ratings.isnull().sum().sum())

# ---- Chart 1: Rating distribution ----
plt.figure(figsize=(7, 5))
sns.countplot(x='rating', data=ratings, color='steelblue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('results/1_rating_distribution.png', dpi=150)
plt.close()

# ---- Chart 2: Long tail ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ratings.groupby('user_id').size().hist(bins=50, ax=ax[0], color='teal')
ax[0].set_yscale('log')
ax[0].set_title('Ratings per User')
ax[0].set_xlabel('Number of ratings')
ratings.groupby('movie_id').size().hist(bins=50, ax=ax[1], color='coral')
ax[1].set_yscale('log')
ax[1].set_title('Ratings per Movie')
ax[1].set_xlabel('Number of ratings')
plt.tight_layout()
plt.savefig('results/2_long_tail.png', dpi=150)
plt.close()

# ---- Chart 3: Average rating per genre ----
df = ratings.merge(movies, on='movie_id')
df['genre'] = df['genres'].str.split('|')
genre_df = df.explode('genre')
genre_avg = genre_df.groupby('genre')['rating'].mean().sort_values()
plt.figure(figsize=(8, 7))
genre_avg.plot(kind='barh', color='mediumpurple')
plt.title('Average Rating per Genre')
plt.xlabel('Average rating')
plt.xlim(3, 4.2)
plt.tight_layout()
plt.savefig('results/3_genre_ratings.png', dpi=150)
plt.close()

# ---- Chart 4: Sparsity ----
n_users = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()
sparsity = 1 - len(ratings) / (n_users * n_movies)
print(f"Sparsity: {sparsity:.2%}")
plt.figure(figsize=(6, 5))
plt.bar(['Known ratings', 'Missing ratings'], [1 - sparsity, sparsity],
        color=['seagreen', 'lightgray'])
plt.title(f'Rating Matrix Sparsity: {sparsity:.1%} empty')
plt.ylabel('Share of matrix')
plt.tight_layout()
plt.savefig('results/4_sparsity.png', dpi=150)
plt.close()

# ---- Top movies (at least 100 ratings) ----
stats = ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
stats = stats[stats['count'] >= 100].merge(movies, on='movie_id')
print("\nTop 10 highest-rated movies (100+ ratings):")
print(stats.sort_values('mean', ascending=False).head(10)[['title', 'mean', 'count']])

print("\nDone! Charts saved in the results folder.")