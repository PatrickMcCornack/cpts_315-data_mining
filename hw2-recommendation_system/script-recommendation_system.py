# -*- coding: utf-8 -*-
"""
The following generates a list of 5 movie recommendations per user.

Adapted from code provided in Panopto Lectures
10/16/2022
"""

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

movies = pd.read_csv('./data/movies.csv')
links = pd.read_csv('./data/links.csv')
tags = pd.read_csv('./data/tags.csv')
ratings = pd.read_csv('./data/ratings.csv')

df = pd.merge(ratings, movies, on="movieId")

# a) Construct a profile of each item. The following is a matrix of user ratings for each movie
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')  

# b) Compute the pearson similarity score for all item-item pairs
# Pearson correlation is the centered cosine similarity
corr_matrix = movie_matrix.corr(method='pearson', min_periods=50)

N = 5  # Neighborhood set size
new_file = ''
for i in range(0, len(movie_matrix)):  # For each user
    userratings = movie_matrix.iloc[i].dropna()  # Ratings of the given user
    recommend = pd.Series()
    
    for j in range(0, len(userratings)):  # For each movie
      
        similar = corr_matrix[userratings.index[j]].dropna()  # correlation for every non-na movie to movie j
        
        # c) Compute the neighborhood set
        similar.sort_values(inplace = True, ascending = False)
        neighbors = similar[~similar.index.isin(userratings.index)]  # remove items if they've already been rated by user i
        
        if len(neighbors) > N:
            neighbors = similar[0:N]  # Select the 5 most similar movies to this movie (excluding itself) if there are more than N items in list
                   
        # d) Use user i's rating for movie j to estimate the ratings of other users using item similarity 
        neighbors = neighbors.map(lambda x: x * userratings[j])  # Multiply each neighbor by the rating user i gave movie j
                    
        recommend = recommend.append(neighbors)  # Append estimated ratings of 5 most similar movies to movie j
    
    # e) Compute the top 5 recommended movies for the user
    if(len(similar) != 0):    
   
        recommend.sort_values(inplace = True, ascending = False)
        
        new_file += str(i)
        if(len(recommend) > 5):
            for i in range(0, 5):
                movie = movies.loc[movies['title'] == recommend.index[i]]
                new_file += " " + str(int(movie.iloc[:,0]))
            new_file += '\n'
        else:
            for i in range(0, len(recommend)):
                movie = movies.loc[movies['title'] == recommend.index[i]]
                new_file += " " + str(int(movie.iloc[:,0]))
            new_file += '\n'

with open("output.txt", "w+") as f:
    f.write(str(new_file))    
        
        

    
    
    
    