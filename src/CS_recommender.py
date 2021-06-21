import pandas as pd
import numpy as np
import sys
import math

def subtract_mean(ratings):
    mean_subtracted_ratings = np.zeros_like(ratings)
    
    for i in range(ratings.shape[0]):
        nonzero_idx = ratings[i].nonzero()[0]
        sum_ratings = np.sum(ratings[i]) 
        num_nonzero = len(nonzero_idx)
        
        if num_nonzero == 0:                
            avg_rating = 0
        else :
            avg_rating = sum_ratings / num_nonzero
        mean_subtracted_ratings[i, nonzero_idx] = ratings[i, nonzero_idx] - avg_rating 

    return mean_subtracted_ratings

def collaborative_filtering(ratings):
    similarity = np.zeros((ratings.shape[0], ratings.shape[0]))
    num_r, num_c = ratings.shape
    
    for i in range (num_r):
        for j in range(i, num_r):
            sum_i = 0 
            sum_j = 0
            dot_product = 0
            for k in range(num_c):
                if ratings[i,k] != 0 and ratings[j,k] != 0:
                    sum_i += ratings[i,k]**2
                    sum_j += ratings[j,k]**2
                    dot_product += ratings[i,k] * ratings[j,k]
                
            if dot_product != 0: 
                similarity[i,j] = dot_product / (math.sqrt(sum_i) * math.sqrt(sum_j))
                similarity[j,i] = similarity[i,j]
    return similarity

def predict(ratings, similarity, k=10):
    pred = np.zeros(ratings.shape)
    
    for u in range(ratings.shape[0]):
        for i in range(ratings.shape[1]):
            watched_i = ratings[:,i].nonzero()[0]
            if u in watched_i:
                watched_i = np.setdiff1d(watched_i, u)
                
            similarity_u = similarity[u, watched_i]
            similar_idx = np.argsort(similarity_u)[::-1][:k]
            similar_idx = similar_idx[np.where(similarity_u[similar_idx] > 0)]
            similar_idx = watched_i[similar_idx]
            sum_similarity = np.sum(similarity[u, similar_idx])
            
            if sum_similarity == 0:
                sum_similarity = 1
            pred[u][i] = np.sum(similarity[u, similar_idx] * ratings[similar_idx, i]) / sum_similarity
    return pred

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    
    print("Load Data ... ", end="", flush=True)
    df_ratings = pd.read_csv("./data-2/"+train, sep='\s+', names=['user','movie','rating','timestamp'])
    df_ratings.drop('timestamp', axis=1, inplace=True)
    df_test = pd.read_csv("./data-2/"+test, sep='\s+', names=['user','movie','rating','timestamp'])
    
    n_users = max(max(df_ratings.user.unique()),max(df_test.user.unique()))
    n_items = max(max(df_ratings.movie.unique()),max(df_test.movie.unique()))
    user_movie = np.zeros((n_users+1, n_items+1)) 
    for row in df_ratings.itertuples(index=False):
        user_id, movie_id, _ = row
        user_movie[user_id, movie_id] = row[2]
    mean_user_movie = subtract_mean(user_movie)
    print("Done")
    
    print("Calculate similarity ... ", end="", flush=True)
    similarity = collaborative_filtering(mean_user_movie)
    print("Done")
    
    print("Calcualte predict ratings ... ", end="", flush=True)
    predict_ratings = predict(user_movie, similarity)
    print("Done")
    
    print("Test ... ", end="", flush=True)
    result_df = df_test.drop(['timestamp'], axis=1).copy()
    result_df.astype({'user':'int','movie':'int','rating':'float'})
    for idx, row in result_df.iterrows():
        user_idx = row['user']
        movie_idx = row['movie']
        result_df.loc[idx,'rating'] = predict_ratings[user_idx][movie_idx]
     
    result_df.to_csv('./test/'+train[:2]+".base_prediction.txt", sep='\t', index=False, header=False)
    print("Done")
    