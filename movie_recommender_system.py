"""
==============================================================================
CONTENT-BASED MOVIE RECOMMENDER SYSTEM
==============================================================================
Assignment: Build a recommendation system based on movie content (genre, overview)
Author: Student Submission
Date: 2026
==============================================================================
"""

# ============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 2. CREATE SAMPLE MOVIE DATASET
# ============================================================================

# Creating a comprehensive movie dataset with real movie information
movies_data = {
    'movie_id': range(1, 26),
    
    'title': [
        'The Avengers', 'Iron Man', 'Captain America', 'Thor', 'Black Widow',
        'Titanic', 'The Notebook', 'La La Land', 'Pride and Prejudice', 'Romeo and Juliet',
        'Inception', 'Interstellar', 'The Matrix', 'Blade Runner 2049', 'Ex Machina',
        'The Dark Knight', 'Joker', 'Batman Begins', 'The Godfather', 'Scarface',
        'Toy Story', 'Finding Nemo', 'The Lion King', 'Frozen', 'Moana'
    ],
    
    'genre': [
        'Action, Superhero, Adventure', 'Action, Superhero, Sci-Fi', 'Action, Superhero, War', 
        'Action, Superhero, Fantasy', 'Action, Spy, Thriller',
        
        'Romance, Drama, Disaster', 'Romance, Drama', 'Romance, Musical, Drama', 
        'Romance, Drama, Period', 'Romance, Drama, Tragedy',
        
        'Sci-Fi, Thriller, Action', 'Sci-Fi, Drama, Adventure', 'Sci-Fi, Action, Cyberpunk', 
        'Sci-Fi, Mystery, Thriller', 'Sci-Fi, Thriller, Drama',
        
        'Action, Crime, Thriller', 'Crime, Drama, Thriller', 'Action, Crime, Superhero', 
        'Crime, Drama, Mafia', 'Crime, Drama, Gangster',
        
        'Animation, Family, Comedy', 'Animation, Family, Adventure', 'Animation, Family, Drama', 
        'Animation, Family, Musical', 'Animation, Family, Adventure'
    ],
    
    'overview': [
        'Earth mightiest heroes must come together to stop an alien invasion led by Loki and save the planet from destruction.',
        'Genius billionaire Tony Stark builds a high-tech armored suit to fight evil and protect the world after being captured by terrorists.',
        'Steve Rogers volunteers for a secret experiment that transforms him into Captain America, a super soldier fighting against evil forces.',
        'The mighty Thor is cast out of Asgard and must prove himself worthy to reclaim his power and save both Earth and Asgard.',
        'Natasha Romanoff confronts her past as a Russian spy while dealing with a dangerous conspiracy that threatens her family.',
        
        'A young aristocrat falls in love with a poor artist aboard the ill-fated maiden voyage of the RMS Titanic.',
        'A poor young man falls in love with a rich young woman giving them one summer together before circumstances force them apart.',
        'An aspiring actress and a dedicated jazz musician struggle to make their dreams come true while falling in love in Los Angeles.',
        'Elizabeth Bennet must overcome her prejudice against the proud Mr. Darcy to find true love in 19th century England.',
        'Two young star-crossed lovers from feuding families fall deeply in love leading to tragic consequences.',
        
        'A thief who steals corporate secrets through dream-sharing technology is given a final chance at redemption involving planting an idea.',
        'A team of explorers travel through a wormhole in space to ensure humanity survival by finding a new habitable planet.',
        'A computer hacker discovers that reality as he knows it is actually a simulated world created by machines.',
        'A young blade runner discovers a secret that could plunge society into chaos and lead him to find a legendary blade runner.',
        'A young programmer is selected to evaluate the human qualities of a highly advanced humanoid artificial intelligence.',
        
        'Batman faces the Joker, a criminal mastermind who wants to plunge Gotham City into anarchy and chaos.',
        'A failed comedian descends into madness and becomes the criminal mastermind known as the Joker in Gotham City.',
        'Bruce Wayne becomes Batman to fight crime and corruption in Gotham City after witnessing his parents murder.',
        'The aging patriarch of an organized crime dynasty transfers control to his reluctant son leading to violence.',
        'A Cuban refugee becomes a powerful drug lord in Miami during the cocaine boom of the 1980s.',
        
        'Toys come to life when humans are not around and must work together to find their way back home.',
        'A clownfish embarks on a journey across the ocean to find his son who was captured by divers.',
        'A young lion prince flees his kingdom after his father is killed but must return to take his rightful place.',
        'A fearless princess teams up with a rugged iceman to find her sister whose icy powers have trapped the kingdom in eternal winter.',
        'A spirited teenager sails out on a daring mission to save her people guided by the ocean and a legendary demigod.'
    ],
    
    'poster_url': [
        'https://image.tmdb.org/t/p/w500/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg',
        'https://image.tmdb.org/t/p/w500/78lPtwv72eTNqFW9COBYI0dWDJa.jpg',
        'https://image.tmdb.org/t/p/w500/vSNxAJTlD0r02V9sPYpOjqDZXUK.jpg',
        'https://image.tmdb.org/t/p/w500/prSfAi1xGrhLQNxVSUFh61xQ4Qy.jpg',
        'https://image.tmdb.org/t/p/w500/qAZ0pzat24kLdO3o8ejmbLxyOac.jpg',
        
        'https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg',
        'https://image.tmdb.org/t/p/w500/rNzQyW4f8B8cQeg7Dgj3n6eT5k9.jpg',
        'https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg',
        'https://image.tmdb.org/t/p/w500/sGjIvtVvTlWnia2zfJfHz81pZ9Q.jpg',
        'https://image.tmdb.org/t/p/w500/xBKGJQsAIeweesB79KC89FpBrVr.jpg',
        
        'https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg',
        'https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg',
        'https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg',
        'https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg',
        'https://image.tmdb.org/t/p/w500/qvktm0BHcnmDpul4Hz01GIazWPr.jpg',
        
        'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
        'https://image.tmdb.org/t/p/w500/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg',
        'https://image.tmdb.org/t/p/w500/dr6x4GyyeggnhUKHkj5YYAVQHUM.jpg',
        'https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg',
        'https://image.tmdb.org/t/p/w500/iQ0yzm9gfR2vEZZ3JNhJRGr1b3I.jpg',
        
        'https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg',
        'https://image.tmdb.org/t/p/w500/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg',
        'https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg',
        'https://image.tmdb.org/t/p/w500/kgwjIb2JDHRhNk13lmSxiClFjc4.jpg',
        'https://image.tmdb.org/t/p/w500/4JeejGugONWpJkbnvL12hVoYEDa.jpg'
    ]
}

# Create DataFrame
df = pd.DataFrame(movies_data)

print("="*80)
print("MOVIE DATASET LOADED")
print("="*80)
print(f"Total Movies: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 Movies:")
print(df[['movie_id', 'title', 'genre']].head())
print("="*80)

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("STEP 1: DATA PREPROCESSING")
print("="*80)

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Handle missing values (fill NaN with empty strings)
df['genre'] = df['genre'].fillna('')
df['overview'] = df['overview'].fillna('')

# Combine relevant text features (genre + overview)
# This creates a comprehensive content description for each movie
df['content'] = df['genre'] + ' ' + df['overview']

# Clean text: convert to lowercase
df['content'] = df['content'].str.lower()

print("\n✓ Missing values handled")
print("✓ Combined 'genre' and 'overview' into 'content' column")
print("✓ Text converted to lowercase")

print("\nSample content for first movie:")
print(f"Movie: {df.iloc[0]['title']}")
print(f"Content: {df.iloc[0]['content'][:150]}...")
print("="*80)

# ============================================================================
# 4. FEATURE ENGINEERING - TF-IDF VECTORIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

print("\n🔍 USING: TF-IDF Vectorizer")
print("\n📝 WHY TF-IDF?")
print("""
TF-IDF (Term Frequency-Inverse Document Frequency) is chosen because:

1. HANDLES IMPORTANCE: TF-IDF gives higher weight to unique/important words
   and lower weight to common words (like 'the', 'is', 'and')

2. BETTER THAN COUNT: Unlike CountVectorizer which just counts word frequency,
   TF-IDF considers how rare/common a word is across all documents

3. IDEAL FOR TEXT SIMILARITY: TF-IDF captures the semantic importance of words,
   making it perfect for finding similar content

4. REDUCES NOISE: Common words that appear in many movies get lower scores,
   focusing on distinctive features

Example: Word 'superhero' in a superhero movie gets HIGH TF-IDF score (rare & important)
         Word 'the' gets LOW TF-IDF score (common & not distinctive)
""")

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=1000,      # Limit to top 1000 features
    stop_words='english',   # Remove common English words
    ngram_range=(1, 2)      # Use both single words and two-word phrases
)

# Fit and transform the content
tfidf_matrix = tfidf.fit_transform(df['content'])

print(f"✓ TF-IDF Matrix Shape: {tfidf_matrix.shape}")
print(f"  - {tfidf_matrix.shape[0]} movies")
print(f"  - {tfidf_matrix.shape[1]} features (words/phrases)")
print("="*80)

# ============================================================================
# 5. SIMILARITY COMPUTATION - COSINE SIMILARITY
# ============================================================================

print("\n" + "="*80)
print("STEP 3: COMPUTING SIMILARITY MATRIX")
print("="*80)

print("\n🔍 USING: Cosine Similarity")
print("\n📝 WHY COSINE SIMILARITY?")
print("""
Cosine Similarity is the best choice because:

1. MEASURES DIRECTION, NOT MAGNITUDE: It measures the angle between vectors,
   not their length. This means documents of different lengths can still be similar.

2. SCALE-INVARIANT: A short movie overview and a long overview can be equally similar
   if they talk about the same topics.

3. RANGE 0-1: Easy to interpret
   - 1.0 = Identical content
   - 0.5 = Moderately similar
   - 0.0 = Completely different

4. PERFECT FOR TEXT: Works excellently with TF-IDF vectors for content comparison

Formula: similarity = (A · B) / (||A|| × ||B||)
""")

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"✓ Similarity Matrix Shape: {cosine_sim.shape}")
print(f"  - {cosine_sim.shape[0]} x {cosine_sim.shape[1]} matrix")
print(f"  - Each cell represents similarity between two movies")

print("\nSample similarity scores (The Avengers vs other movies):")
avengers_idx = 0
for i in range(5):
    print(f"  {df.iloc[avengers_idx]['title']} ↔ {df.iloc[i]['title']}: {cosine_sim[avengers_idx][i]:.4f}")
print("="*80)

# ============================================================================
# 6. RECOMMENDATION FUNCTION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CREATING RECOMMENDATION FUNCTION")
print("="*80)

def recommend(movie_title, top_n=5):
    """
    Recommend similar movies based on content similarity
    
    Parameters:
    -----------
    movie_title : str
        Title of the movie to get recommendations for
    top_n : int, default=5
        Number of recommendations to return
    
    Returns:
    --------
    DataFrame with recommended movies, their similarity scores, and poster URLs
    """
    
    # Check if movie exists in dataset
    if movie_title not in df['title'].values:
        print(f"❌ Movie '{movie_title}' not found in database!")
        print("\nAvailable movies:")
        for title in df['title'].values:
            print(f"  - {title}")
        return None
    
    # Find the index of the movie
    movie_idx = df[df['title'] == movie_title].index[0]
    
    # Get similarity scores for this movie with all other movies
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))
    
    # Sort movies by similarity score (descending order)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies (excluding the movie itself - index 0)
    # Index 0 will always be the movie itself with similarity = 1.0
    similar_movies_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    similar_movies_scores = [i[1] for i in similarity_scores[1:top_n+1]]
    
    # Create recommendations DataFrame
    recommendations = df.iloc[similar_movies_indices][['title', 'genre', 'poster_url']].copy()
    recommendations['similarity_score'] = similar_movies_scores
    
    # Reorder columns
    recommendations = recommendations[['title', 'genre', 'similarity_score', 'poster_url']]
    
    return recommendations

print("\n✓ Recommendation function created: recommend(movie_title, top_n=5)")
print("\nFunction Logic:")
print("  1. Find index of input movie")
print("  2. Get similarity scores with all movies")
print("  3. Sort by similarity (descending)")
print("  4. Exclude the movie itself")
print("  5. Return top N similar movies")
print("="*80)

# ============================================================================
# 7. TESTING THE RECOMMENDATION SYSTEM
# ============================================================================

print("\n" + "="*80)
print("STEP 5: TESTING THE RECOMMENDER SYSTEM")
print("="*80)

# Test 1: Recommend movies similar to "The Avengers"
print("\n🎬 TEST 1: Movies similar to 'The Avengers'")
print("-" * 80)
recommendations_1 = recommend("The Avengers", top_n=5)
print(f"\nInput Movie: The Avengers (Action, Superhero)")
print(f"\nTop 5 Recommendations:\n")
for idx, row in recommendations_1.iterrows():
    print(f"{idx+1}. {row['title']}")
    print(f"   Genre: {row['genre']}")
    print(f"   Similarity Score: {row['similarity_score']:.4f}")
    print(f"   Poster: {row['poster_url']}")
    print()

# Test 2: Recommend movies similar to "Titanic"
print("\n" + "="*80)
print("🎬 TEST 2: Movies similar to 'Titanic'")
print("-" * 80)
recommendations_2 = recommend("Titanic", top_n=5)
print(f"\nInput Movie: Titanic (Romance, Drama)")
print(f"\nTop 5 Recommendations:\n")
for idx, row in recommendations_2.iterrows():
    print(f"{idx+1}. {row['title']}")
    print(f"   Genre: {row['genre']}")
    print(f"   Similarity Score: {row['similarity_score']:.4f}")
    print(f"   Poster: {row['poster_url']}")
    print()

# Test 3: Recommend movies similar to "Inception"
print("\n" + "="*80)
print("🎬 TEST 3: Movies similar to 'Inception'")
print("-" * 80)
recommendations_3 = recommend("Inception", top_n=5)
print(f"\nInput Movie: Inception (Sci-Fi, Thriller)")
print(f"\nTop 5 Recommendations:\n")
for idx, row in recommendations_3.iterrows():
    print(f"{idx+1}. {row['title']}")
    print(f"   Genre: {row['genre']}")
    print(f"   Similarity Score: {row['similarity_score']:.4f}")
    print(f"   Poster: {row['poster_url']}")
    print()

# ============================================================================
# 8. EXPLANATIONS AND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: EXPLANATIONS AND ANALYSIS")
print("="*80)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    CONTENT-BASED VS COLLABORATIVE FILTERING               ║
╚══════════════════════════════════════════════════════════════════════════╝

📊 CONTENT-BASED FILTERING (What we built):
   └─ Uses: Item features (genre, overview, description)
   └─ Logic: "This movie is similar to movies you liked"
   └─ Example: If you liked "The Avengers" → Recommend "Iron Man" 
              (both are superhero action movies)
   └─ Data Needed: Only item metadata (no user ratings needed)

👥 COLLABORATIVE FILTERING (Different approach):
   └─ Uses: User ratings and behavior
   └─ Logic: "Users who liked this also liked that"
   └─ Example: If User A and User B both liked Movie X,
              and User A liked Movie Y → Recommend Movie Y to User B
   └─ Data Needed: User-item interaction matrix (ratings, views)

KEY DIFFERENCES:
┌─────────────────────┬──────────────────────┬─────────────────────────┐
│ Aspect              │ Content-Based        │ Collaborative           │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Uses                │ Item features        │ User ratings            │
│ Cold Start Problem  │ No (for items)       │ Yes (severe)            │
│ Diversity           │ Low (similar items)  │ High (diverse items)    │
│ New Items           │ Easy to handle       │ Hard (no ratings yet)   │
│ New Users           │ Hard (no history)    │ Hard (no ratings)       │
│ Explanation         │ Easy (feature-based) │ Hard (pattern-based)    │
└─────────────────────┴──────────────────────┴─────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════╗
║                        LIMITATION OF CONTENT-BASED                        ║
╚══════════════════════════════════════════════════════════════════════════╝

⚠️  MAJOR LIMITATION: "Filter Bubble" / Over-Specialization

PROBLEM:
   The system only recommends movies very similar to what you've watched.
   It creates an "echo chamber" where you never discover different genres.

EXAMPLE:
   - You watch superhero movies → System only recommends superhero movies
   - You NEVER discover that you might love sci-fi, romance, or animation
   - Your recommendations become too predictable and boring

REAL-WORLD IMPACT:
   ❌ No serendipitous discoveries (happy accidents)
   ❌ Limited exposure to diverse content
   ❌ User gets stuck in same genre forever
   ❌ Cannot capture changing user preferences

ANALOGY:
   It's like having a friend who ONLY talks about topics you already know.
   You never learn anything new or explore different interests!

╔══════════════════════════════════════════════════════════════════════════╗
║                          POSSIBLE IMPROVEMENTS                            ║
╚══════════════════════════════════════════════════════════════════════════╝

🚀 IMPROVEMENT: Hybrid Recommender System

SOLUTION:
   Combine Content-Based + Collaborative Filtering

HOW IT WORKS:
   1. Content-Based: Find similar movies based on features
   2. Collaborative: Find movies liked by similar users
   3. HYBRID: Weighted average of both approaches
   
FORMULA:
   Final_Score = (0.7 × Content_Score) + (0.3 × Collaborative_Score)

BENEFITS:
   ✅ Solves over-specialization (collaborative adds diversity)
   ✅ Handles cold start better (content helps with new items)
   ✅ More accurate recommendations (best of both worlds)
   ✅ Balances similarity with serendipity

EXAMPLE OUTPUT:
   For someone who watched "The Avengers":
   
   Content-Based might give: Iron Man, Thor, Captain America (all superhero)
   Collaborative adds: Inception, Interstellar (other users also liked these)
   HYBRID gives: Mix of similar superhero movies + surprising good matches!

OTHER IMPROVEMENTS:
   • Add user ratings as a feature
   • Include cast, director, year as features
   • Use deep learning (neural collaborative filtering)
   • Add diversity penalty (force different genres)
   • Temporal dynamics (trending movies get boost)
""")

print("="*80)

# ============================================================================
# 9. SAVE RESULTS TO CSV
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save full dataset with content features
df.to_csv('movies_dataset.csv', index=False)
print("✓ Full dataset saved: movies_dataset.csv")

# Save example recommendations
recommendations_1.to_csv('recommendations_avengers.csv', index=False)
print("✓ Recommendations for 'The Avengers' saved: recommendations_avengers.csv")

print("\n" + "="*80)
print("✅ MOVIE RECOMMENDER SYSTEM COMPLETED SUCCESSFULLY!")
print("="*80)

print("""
📌 SUMMARY:
   ✓ Dataset: 25 movies with genre and overview
   ✓ Preprocessing: Combined text features, cleaned data
   ✓ Vectorization: TF-IDF with 1000 features
   ✓ Similarity: Cosine similarity matrix (25x25)
   ✓ Function: recommend() returns top 5 similar movies
   ✓ Testing: Tested with 3 different movies
   ✓ Explanations: Content vs Collaborative, Limitations, Improvements

🎯 ASSIGNMENT REQUIREMENTS MET:
   ✅ Data preprocessing (missing values, text cleaning)
   ✅ Feature engineering (TF-IDF chosen with explanation)
   ✅ Similarity computation (Cosine similarity with explanation)
   ✅ Recommendation function (working and tested)
   ✅ Evaluation & explanation (all questions answered)
   ✅ Code in text format with examples
   ✅ Movie images (poster URLs) included
""")

print("\n" + "="*80)
print("END OF PROGRAM")
print("="*80)
