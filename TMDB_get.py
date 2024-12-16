import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

#TMDB API keys
API = os.getenv('API_TMDB')
AUTH = os.getenv('AUTH_TMDB')

# Get list of movie genres from TMDB
url = "https://api.themoviedb.org/3/genre/movie/list?"
headers = {
    "accept": "application/json",
    "Authorization": "Bearer " + AUTH,
    "Language": "en"
}
genres_raw = requests.get(url, headers=headers).json()['genres']

# Turn genres from TMDB into a dictionary
genres = {}
for genre in genres_raw:
    genres[genre['id']] = genre['name']

#Set up for getting list of movies
data = {
        'genre_ids': [],
        'poster_path': [],
        'name': []}
page = 1

start_yr = '1990-01-01'
end_yr = '1999-12-15'
filename = 're2'
num = 0

# Goes through 500 pages of movies on TMDB and saves each movie's genres and posters
while page <= 5:
    try:
        url = f'https://api.themoviedb.org/3/discover/movie?api_key={API}&language=en-US&sort_by=release_date.asc&page={page}&primary_release_date.gte={start_yr}&primary_release_date.lte={end_yr}'
        # url = f'https://api.themoviedb.org/3/movie/popular?api_key={API}&language=en&page={page}'
        movies = requests.get(url).json()
        
        for movie in movies['results']:
            #Preprocess: Ensure movies with no list genre or posters are removed
            if movie['genre_ids'] != [] and movie['poster_path'] != None:

                #Preprocess: genre id values into its respective genres
                movie_genres = []
                for genre_id in movie['genre_ids']:
                    movie_genres.append(genres[genre_id])
                
                data['genre_ids'].append(movie_genres)
                data['poster_path'].append(movie['poster_path'])
                data['name'].append(movie['original_title'])
    except:
        break

    page += 1

#Save as csv
data_df = pd.DataFrame(data)
data_df.to_csv(f"data/raw_{filename}.csv")

# data = pd.read_csv("data/raw_1970s.csv")

#Download all poster images and name them by number in the order of the csv file.
# url = "https://image.tmdb.org/t/p/original"
# for idx in range(len(data['poster_path'])):
#         findImg = requests.get(url + str(data['poster_path'][idx])).content #Find poster from online
        
#         #Download poster
#         img = open("data/posters/" + str(idx+num) + '.jpg', 'wb') 
#         img.write(findImg)
#         img.close()

# r60s = pd.read_csv("data/raw_1960s.csv")
# r70s = pd.read_csv("data/raw_1970s.csv")
# r80s = pd.read_csv("data/raw_1980s.csv")
# r90s = pd.read_csv("data/raw_1990s.csv")
# r00s = pd.read_csv("data/raw_2000s.csv")
# r10s = pd.read_csv("data/raw_2010s.csv")
# r20s = pd.read_csv("data/raw_2020s.csv")

# r70s['Unnamed: 0'] = r70s['Unnamed: 0'].apply(lambda x: int(x) + 6299)
# r80s['Unnamed: 0'] = r80s['Unnamed: 0'].apply(lambda x: int(x) + 12280)
# r90s['Unnamed: 0'] = r90s['Unnamed: 0'].apply(lambda x: int(x) + 18349)
# r00s['Unnamed: 0'] = r00s['Unnamed: 0'].apply(lambda x: int(x) + 24229)
# r10s['Unnamed: 0'] = r10s['Unnamed: 0'].apply(lambda x: int(x) + 29844)
# r20s['Unnamed: 0'] = r20s['Unnamed: 0'].apply(lambda x: int(x) + 35452)

# raw = pd.concat([r60s, r70s, r80s, r90s, r00s, r10s, r20s])
# raw.to_csv(f"data/raw.csv")
