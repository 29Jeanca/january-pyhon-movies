import pandas as pd
from sklearn.neighbors import NearestNeighbors

ratingMovies = pd.read_csv('ratings.csv')

knn= NearestNeighbors(metric='cosine',algorithm='brute')

knn.fit(ratingMovies)

userId = 1

distancias, indices = knn.kneighbors(ratingMovies.iloc[userId - 1, :].values.reshape(1, -1), n_neighbors=10)

for i in range(0,len(distancias.flatten())):
        if i == 0:
                print('Las peliculas recomendadas para el usuario son: {}'.format(userId))
        else:
                print('{}: {}'.format(i, ratingMovies.index[indices.flatten()[i]]))