"""
============================================================
  INDIAN MOVIES DATASET GENERATOR
  Bollywood + South + Hollywood movies
  With realistic ratings data
============================================================
"""

import pandas as pd
import numpy as np

def create_indian_movies_dataset():
    """
    Real Indian + Hollywood movies ka dataset banata hai.
    Genres, tags aur ratings ke saath.
    """

    movies_data = [
        # ── BOLLYWOOD ACTION / WAR ──────────────────────────────────────
        {"movieId": 1,  "title": "Uri: The Surgical Strike",     "genres": "Action|War|Patriotic",         "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.8, "rating_count": 4500, "tags": "army surgical strike patriotic war india pakistan"},
        {"movieId": 2,  "title": "Dhurandhar",                   "genres": "Action|Thriller|Patriotic",    "year": 2024, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.5, "rating_count": 3200, "tags": "action thriller patriotic mission spy"},
        {"movieId": 3,  "title": "War",                          "genres": "Action|Thriller|Spy",          "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.3, "rating_count": 4200, "tags": "action spy thriller mission hrithik"},
        {"movieId": 4,  "title": "Pathaan",                      "genres": "Action|Spy|Thriller",          "year": 2023, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.2, "rating_count": 4800, "tags": "spy action thriller shahrukh mission india"},
        {"movieId": 5,  "title": "Tiger Zinda Hai",              "genres": "Action|Spy|Patriotic",         "year": 2017, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.1, "rating_count": 4000, "tags": "spy action patriotic salman mission"},
        {"movieId": 6,  "title": "Tiger 3",                      "genres": "Action|Spy|Thriller",          "year": 2023, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 3.9, "rating_count": 3500, "tags": "spy action thriller salman mission"},
        {"movieId": 7,  "title": "Raazi",                        "genres": "Spy|Drama|Patriotic",          "year": 2018, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.6, "rating_count": 3800, "tags": "spy drama patriotic alia pakistan undercover"},
        {"movieId": 8,  "title": "Kesari",                       "genres": "Action|War|Historical",        "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.4, "rating_count": 3600, "tags": "war historical action sikh akshay battle"},
        {"movieId": 9,  "title": "Shershah",                     "genres": "War|Drama|Patriotic|Romance",  "year": 2021, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 4100, "tags": "war army kargil patriotic romance sidharth"},
        {"movieId": 10, "title": "Bard of Blood",                "genres": "Spy|Action|Thriller",          "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.0, "rating_count": 2800, "tags": "spy action thriller raw agent mission"},
        {"movieId": 11, "title": "Baby",                         "genres": "Action|Spy|Thriller",          "year": 2015, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.2, "rating_count": 3400, "tags": "spy action thriller akshay counter terrorism"},
        {"movieId": 12, "title": "Holiday: A Soldier Is Never Off Duty", "genres": "Action|Spy|Thriller", "year": 2014, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.0, "rating_count": 2900, "tags": "action spy thriller soldier akshay mission"},

        # ── BOLLYWOOD DRAMA / SOCIAL ────────────────────────────────────
        {"movieId": 13, "title": "Dangal",                       "genres": "Drama|Sports|Biography",       "year": 2016, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.9, "rating_count": 5000, "tags": "wrestling sports biography aamir daughters inspiration"},
        {"movieId": 14, "title": "3 Idiots",                     "genres": "Comedy|Drama|Education",       "year": 2009, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.9, "rating_count": 5500, "tags": "comedy drama education college friends aamir"},
        {"movieId": 15, "title": "Taare Zameen Par",             "genres": "Drama|Family|Education",       "year": 2007, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.8, "rating_count": 4800, "tags": "drama child education dyslexia family aamir"},
        {"movieId": 16, "title": "PK",                           "genres": "Comedy|Drama|SocialCommentary","year": 2014, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 4700, "tags": "comedy drama religion social aamir alien"},
        {"movieId": 17, "title": "Dil Dhadakne Do",              "genres": "Drama|Family|Comedy",          "year": 2015, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.2, "rating_count": 3200, "tags": "family drama comedy travel rich"},
        {"movieId": 18, "title": "Kapoor and Sons",              "genres": "Drama|Family|Romance",         "year": 2016, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.3, "rating_count": 3100, "tags": "family drama romance emotional"},
        {"movieId": 19, "title": "Andhadhun",                    "genres": "Thriller|Mystery|Crime",       "year": 2018, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.8, "rating_count": 4400, "tags": "thriller mystery crime blind pianist"},
        {"movieId": 20, "title": "Drishyam 2",                   "genres": "Thriller|Mystery|Drama",       "year": 2022, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.6, "rating_count": 4200, "tags": "thriller mystery drama family crime ajay"},
        {"movieId": 21, "title": "Article 15",                   "genres": "Drama|Crime|SocialCommentary", "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.5, "rating_count": 3300, "tags": "social drama crime caste discrimination ayushmann"},
        {"movieId": 22, "title": "Swades",                       "genres": "Drama|Patriotic|SocialCommentary","year":2004,"language": "Hindi",  "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 3800, "tags": "patriotic drama social village development shahrukh"},

        # ── BOLLYWOOD ROMANCE / FAMILY ──────────────────────────────────
        {"movieId": 23, "title": "Dilwale Dulhania Le Jayenge",  "genres": "Romance|Drama|Family",         "year": 1995, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.8, "rating_count": 5200, "tags": "romance love classic shahrukh kajol europe"},
        {"movieId": 24, "title": "Kabhi Khushi Kabhie Gham",     "genres": "Drama|Family|Romance",         "year": 2001, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.5, "rating_count": 4500, "tags": "family drama romance rich poor srk hrithik"},
        {"movieId": 25, "title": "Bajrangi Bhaijaan",            "genres": "Drama|Family|Comedy|Patriotic","year": 2015, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 4600, "tags": "family drama comedy pakistan child salman"},
        {"movieId": 26, "title": "Jab We Met",                   "genres": "Romance|Comedy|Drama",         "year": 2007, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.6, "rating_count": 4300, "tags": "romance comedy drama travel kareena shahid"},
        {"movieId": 27, "title": "Queen",                        "genres": "Drama|Comedy|Romance",         "year": 2014, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.6, "rating_count": 3900, "tags": "drama comedy solo travel woman empowerment kangana"},
        {"movieId": 28, "title": "Piku",                         "genres": "Drama|Comedy|Family",          "year": 2015, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.4, "rating_count": 3500, "tags": "family comedy drama road trip deepika amitabh"},

        # ── BOLLYWOOD COMEDY ────────────────────────────────────────────
        {"movieId": 29, "title": "Golmaal Returns",              "genres": "Comedy|Action|Family",         "year": 2008, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.0, "rating_count": 3200, "tags": "comedy action family ajay rohit shetty"},
        {"movieId": 30, "title": "Hera Pheri",                   "genres": "Comedy|Crime|Drama",           "year": 2000, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.8, "rating_count": 5000, "tags": "comedy crime classic akshay suniel paresh"},
        {"movieId": 31, "title": "Andaz Apna Apna",              "genres": "Comedy|Romance|Family",        "year": 1994, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 4400, "tags": "comedy romance classic aamir salman"},
        {"movieId": 32, "title": "Chup Chup Ke",                 "genres": "Comedy|Drama|Romance",         "year": 2006, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.1, "rating_count": 2900, "tags": "comedy drama romance"},

        # ── SOUTH INDIAN MOVIES ─────────────────────────────────────────
        {"movieId": 33, "title": "RRR",                          "genres": "Action|Adventure|Historical",  "year": 2022, "language": "Telugu",  "industry": "South",     "avg_rating": 4.9, "rating_count": 5500, "tags": "action adventure historical freedom fighters rajamouli"},
        {"movieId": 34, "title": "Baahubali: The Beginning",     "genres": "Action|Adventure|Fantasy",     "year": 2015, "language": "Telugu",  "industry": "South",     "avg_rating": 4.8, "rating_count": 5200, "tags": "action adventure fantasy historical rajamouli epic"},
        {"movieId": 35, "title": "Baahubali 2: The Conclusion",  "genres": "Action|Adventure|Drama",       "year": 2017, "language": "Telugu",  "industry": "South",     "avg_rating": 4.9, "rating_count": 5800, "tags": "action adventure drama historical epic rajamouli"},
        {"movieId": 36, "title": "KGF Chapter 1",                "genres": "Action|Crime|Drama",           "year": 2018, "language": "Kannada", "industry": "South",     "avg_rating": 4.7, "rating_count": 5000, "tags": "action crime drama gold mines yash powerful"},
        {"movieId": 37, "title": "KGF Chapter 2",                "genres": "Action|Crime|Drama",           "year": 2022, "language": "Kannada", "industry": "South",     "avg_rating": 4.8, "rating_count": 5500, "tags": "action crime drama powerful yash sanjay dutt"},
        {"movieId": 38, "title": "Pushpa: The Rise",             "genres": "Action|Crime|Drama",           "year": 2021, "language": "Telugu",  "industry": "South",     "avg_rating": 4.6, "rating_count": 4800, "tags": "action crime drama smuggling allu arjun red sandalwood"},
        {"movieId": 39, "title": "Pushpa 2: The Rule",           "genres": "Action|Crime|Drama",           "year": 2024, "language": "Telugu",  "industry": "South",     "avg_rating": 4.8, "rating_count": 5200, "tags": "action crime drama smuggling allu arjun powerful"},
        {"movieId": 40, "title": "Vikram",                       "genres": "Action|Thriller|Crime",        "year": 2022, "language": "Tamil",   "industry": "South",     "avg_rating": 4.7, "rating_count": 4600, "tags": "action thriller crime kamal haasan lokesh"},
        {"movieId": 41, "title": "Master",                       "genres": "Action|Drama|Thriller",        "year": 2021, "language": "Tamil",   "industry": "South",     "avg_rating": 4.5, "rating_count": 4200, "tags": "action drama college vijay thalapathy"},
        {"movieId": 42, "title": "Beast",                        "genres": "Action|Thriller|Comedy",       "year": 2022, "language": "Tamil",   "industry": "South",     "avg_rating": 3.8, "rating_count": 3200, "tags": "action thriller comedy vijay thalapathy"},
        {"movieId": 43, "title": "2.0",                          "genres": "Action|SciFi|Fantasy",         "year": 2018, "language": "Tamil",   "industry": "South",     "avg_rating": 4.3, "rating_count": 4000, "tags": "scifi action fantasy robot rajinikanth akshay"},
        {"movieId": 44, "title": "Jailer",                       "genres": "Action|Drama|Crime",           "year": 2023, "language": "Tamil",   "industry": "South",     "avg_rating": 4.4, "rating_count": 4100, "tags": "action drama crime rajinikanth jailer father son"},
        {"movieId": 45, "title": "Leo",                          "genres": "Action|Thriller|Crime",        "year": 2023, "language": "Tamil",   "industry": "South",     "avg_rating": 4.3, "rating_count": 3900, "tags": "action thriller crime vijay thalapathy lokesh"},
        {"movieId": 46, "title": "Salaar",                       "genres": "Action|Crime|Drama",           "year": 2023, "language": "Telugu",  "industry": "South",     "avg_rating": 4.2, "rating_count": 3800, "tags": "action crime drama prabhas prashanth neel"},
        {"movieId": 47, "title": "Kalki 2898 AD",                "genres": "SciFi|Action|Fantasy",         "year": 2024, "language": "Telugu",  "industry": "South",     "avg_rating": 4.5, "rating_count": 4400, "tags": "scifi action fantasy future mythology prabhas deepika"},
        {"movieId": 48, "title": "Soorarai Pottru",              "genres": "Drama|Biography|Inspirational","year": 2020, "language": "Tamil",   "industry": "South",     "avg_rating": 4.7, "rating_count": 4200, "tags": "biography drama inspirational aviation suriya"},
        {"movieId": 49, "title": "Super Deluxe",                 "genres": "Drama|Thriller|Comedy",        "year": 2019, "language": "Tamil",   "industry": "South",     "avg_rating": 4.6, "rating_count": 3700, "tags": "drama thriller comedy vijay sethupathi"},
        {"movieId": 50, "title": "Drishyam",                     "genres": "Thriller|Mystery|Drama",       "year": 2013, "language": "Malayalam","industry": "South",    "avg_rating": 4.8, "rating_count": 4500, "tags": "thriller mystery drama family crime mohanlal"},
        {"movieId": 51, "title": "Premam",                       "genres": "Romance|Drama|Comedy",         "year": 2015, "language": "Malayalam","industry": "South",    "avg_rating": 4.6, "rating_count": 3900, "tags": "romance drama comedy college nivin pauly"},
        {"movieId": 52, "title": "Lucifer",                      "genres": "Action|Crime|Drama",           "year": 2019, "language": "Malayalam","industry": "South",    "avg_rating": 4.4, "rating_count": 3800, "tags": "action crime drama politics mohanlal prithviraj"},

        # ── HOLLYWOOD ACTION ────────────────────────────────────────────
        {"movieId": 53, "title": "Avengers: Endgame",            "genres": "Action|SciFi|Adventure",       "year": 2019, "language": "English", "industry": "Hollywood", "avg_rating": 4.9, "rating_count": 6000, "tags": "superhero action scifi marvel avengers thanos"},
        {"movieId": 54, "title": "Top Gun: Maverick",            "genres": "Action|Drama|Military",        "year": 2022, "language": "English", "industry": "Hollywood", "avg_rating": 4.8, "rating_count": 5200, "tags": "action military fighter jet drama tom cruise navy"},
        {"movieId": 55, "title": "Mission Impossible",           "genres": "Action|Spy|Thriller",          "year": 1996, "language": "English", "industry": "Hollywood", "avg_rating": 4.5, "rating_count": 4800, "tags": "spy action thriller mission ethan hunt tom cruise"},
        {"movieId": 56, "title": "John Wick",                    "genres": "Action|Crime|Thriller",        "year": 2014, "language": "English", "industry": "Hollywood", "avg_rating": 4.7, "rating_count": 5000, "tags": "action crime assassin keanu reeves revenge"},
        {"movieId": 57, "title": "The Dark Knight",              "genres": "Action|Crime|Drama",           "year": 2008, "language": "English", "industry": "Hollywood", "avg_rating": 4.9, "rating_count": 6500, "tags": "superhero batman joker crime drama christopher nolan"},
        {"movieId": 58, "title": "Inception",                    "genres": "SciFi|Thriller|Action",        "year": 2010, "language": "English", "industry": "Hollywood", "avg_rating": 4.9, "rating_count": 6200, "tags": "scifi dream thriller action nolan dicaprio mind"},
        {"movieId": 59, "title": "Interstellar",                 "genres": "SciFi|Drama|Adventure",        "year": 2014, "language": "English", "industry": "Hollywood", "avg_rating": 4.9, "rating_count": 6100, "tags": "scifi space adventure drama nolan time travel"},
        {"movieId": 60, "title": "The Martian",                  "genres": "SciFi|Drama|Adventure",        "year": 2015, "language": "English", "industry": "Hollywood", "avg_rating": 4.7, "rating_count": 4900, "tags": "scifi space survival mars matt damon nasa"},
        {"movieId": 61, "title": "Gladiator",                    "genres": "Action|Drama|Historical",      "year": 2000, "language": "English", "industry": "Hollywood", "avg_rating": 4.8, "rating_count": 5100, "tags": "action historical drama roman empire russell crowe"},
        {"movieId": 62, "title": "300",                          "genres": "Action|Historical|War",        "year": 2006, "language": "English", "industry": "Hollywood", "avg_rating": 4.6, "rating_count": 4700, "tags": "action war historical sparta battle gerard butler"},
        {"movieId": 63, "title": "The Revenant",                 "genres": "Adventure|Drama|Survival",     "year": 2015, "language": "English", "industry": "Hollywood", "avg_rating": 4.6, "rating_count": 4500, "tags": "survival adventure drama wilderness dicaprio revenge"},

        # ── HOLLYWOOD THRILLER / CRIME ──────────────────────────────────
        {"movieId": 64, "title": "Shutter Island",               "genres": "Thriller|Mystery|Drama",       "year": 2010, "language": "English", "industry": "Hollywood", "avg_rating": 4.7, "rating_count": 5000, "tags": "thriller mystery psychological drama dicaprio island"},
        {"movieId": 65, "title": "Se7en",                        "genres": "Crime|Thriller|Mystery",       "year": 1995, "language": "English", "industry": "Hollywood", "avg_rating": 4.8, "rating_count": 5300, "tags": "crime thriller detective mystery brad pitt serial killer"},
        {"movieId": 66, "title": "The Silence of the Lambs",     "genres": "Crime|Thriller|Horror",        "year": 1991, "language": "English", "industry": "Hollywood", "avg_rating": 4.8, "rating_count": 5100, "tags": "crime thriller horror hannibal serial killer fbi"},
        {"movieId": 67, "title": "Prisoners",                    "genres": "Thriller|Crime|Drama",         "year": 2013, "language": "English", "industry": "Hollywood", "avg_rating": 4.7, "rating_count": 4700, "tags": "thriller crime kidnapping drama mystery hugh jackman"},
        {"movieId": 68, "title": "Gone Girl",                    "genres": "Thriller|Mystery|Drama",       "year": 2014, "language": "English", "industry": "Hollywood", "avg_rating": 4.6, "rating_count": 4600, "tags": "thriller mystery marriage drama ben affleck fincher"},

        # ── HOLLYWOOD COMEDY / DRAMA ────────────────────────────────────
        {"movieId": 69, "title": "The Pursuit of Happyness",     "genres": "Drama|Biography|Inspirational","year": 2006, "language": "English", "industry": "Hollywood", "avg_rating": 4.8, "rating_count": 5200, "tags": "biography inspirational drama will smith father son"},
        {"movieId": 70, "title": "Forrest Gump",                 "genres": "Drama|Comedy|Romance",         "year": 1994, "language": "English", "industry": "Hollywood", "avg_rating": 4.9, "rating_count": 6000, "tags": "drama comedy romance classic tom hanks life story"},
        {"movieId": 71, "title": "The Shawshank Redemption",     "genres": "Drama|Crime|Inspirational",    "year": 1994, "language": "English", "industry": "Hollywood", "avg_rating": 4.9, "rating_count": 6800, "tags": "drama crime prison friendship hope inspirational classic"},
        {"movieId": 72, "title": "Schindler's List",             "genres": "Drama|Historical|War",         "year": 1993, "language": "English", "industry": "Hollywood", "avg_rating": 4.9, "rating_count": 5900, "tags": "historical war drama holocaust jews spielberg"},
        {"movieId": 73, "title": "Good Will Hunting",            "genres": "Drama|Romance|Inspirational",  "year": 1997, "language": "English", "industry": "Hollywood", "avg_rating": 4.7, "rating_count": 4900, "tags": "drama inspirational genius math robin williams matt damon"},

        # ── EXTRA BOLLYWOOD ─────────────────────────────────────────────
        {"movieId": 74, "title": "Gangs of Wasseypur",           "genres": "Crime|Drama|Action",           "year": 2012, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 4300, "tags": "crime drama action gangster coal mafia anurag kashyap"},
        {"movieId": 75, "title": "Zindagi Na Milegi Dobara",     "genres": "Drama|Comedy|Adventure",       "year": 2011, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.6, "rating_count": 4100, "tags": "adventure comedy drama friends travel spain hrithik"},
        {"movieId": 76, "title": "Lagaan",                       "genres": "Drama|Sports|Historical",      "year": 2001, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.8, "rating_count": 4600, "tags": "cricket sports historical drama british india aamir"},
        {"movieId": 77, "title": "Rang De Basanti",              "genres": "Drama|Action|Patriotic",       "year": 2006, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 4400, "tags": "patriotic drama youth revolution aamir india"},
        {"movieId": 78, "title": "Mughal-E-Azam",                "genres": "Drama|Historical|Romance",     "year": 1960, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.8, "rating_count": 3900, "tags": "historical drama romance mughal classic dilip"},
        {"movieId": 79, "title": "Gully Boy",                    "genres": "Drama|Music|Inspirational",    "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.4, "rating_count": 3700, "tags": "music rap drama street inspirational ranveer"},
        {"movieId": 80, "title": "Tumbbad",                      "genres": "Horror|Fantasy|Mystery",       "year": 2018, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.7, "rating_count": 3800, "tags": "horror fantasy mystery greed folklore sohum"},
        {"movieId": 81, "title": "Stree",                        "genres": "Horror|Comedy|Mystery",        "year": 2018, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.5, "rating_count": 4000, "tags": "horror comedy ghost mystery small town rajkummar"},
        {"movieId": 82, "title": "Bhediya",                      "genres": "Horror|Comedy|Fantasy",        "year": 2022, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.1, "rating_count": 3200, "tags": "horror comedy fantasy werewolf varun dhawan"},
        {"movieId": 83, "title": "OMG: Oh My God!",              "genres": "Comedy|Drama|SocialCommentary","year": 2012, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.5, "rating_count": 3900, "tags": "comedy drama religion social akshay paresh"},
        {"movieId": 84, "title": "Tashan",                       "genres": "Action|Comedy|Romance",        "year": 2008, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 3.2, "rating_count": 1800, "tags": "action comedy romance akshay kareena"},
        {"movieId": 85, "title": "Mission Mangal",               "genres": "Drama|SciFi|Inspirational",    "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.2, "rating_count": 3300, "tags": "space science isro mars mission akshay women scientists"},
        {"movieId": 86, "title": "Rocket Boys",                  "genres": "Drama|Biography|SciFi",        "year": 2022, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.6, "rating_count": 3500, "tags": "biography science drama homi bhabha vikram sarabhai"},
        {"movieId": 87, "title": "Bhaag Milkha Bhaag",          "genres": "Drama|Sports|Biography",       "year": 2013, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.5, "rating_count": 4000, "tags": "sports biography running athletics farhan akhtar partition"},
        {"movieId": 88, "title": "MS Dhoni: The Untold Story",   "genres": "Drama|Sports|Biography",       "year": 2016, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.6, "rating_count": 4300, "tags": "cricket sports biography dhoni sushant"},
        {"movieId": 89, "title": "83",                           "genres": "Drama|Sports|Historical",      "year": 2021, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.4, "rating_count": 3700, "tags": "cricket world cup 1983 historical sports ranveer"},
        {"movieId": 90, "title": "Kabir Singh",                  "genres": "Drama|Romance|Action",         "year": 2019, "language": "Hindi",   "industry": "Bollywood", "avg_rating": 4.1, "rating_count": 4200, "tags": "drama romance love shahid kapoor medical college"},
    ]

    movies_df = pd.DataFrame(movies_data)

    # ── Generate Ratings ────────────────────────────────────────────────
    np.random.seed(42)
    n_users   = 500
    ratings_list = []

    for movie in movies_data:
        n_ratings = min(movie['rating_count'], 300)
        n_ratings = np.random.randint(n_ratings // 3, n_ratings // 2 + 1)
        n_ratings = min(n_ratings, n_users)

        rated_users = np.random.choice(range(1, n_users + 1),
                                        size=n_ratings, replace=False)
        base = movie['avg_rating']

        for user_id in rated_users:
            noise  = np.random.normal(0, 0.5)
            rating = np.clip(base + noise, 1.0, 5.0)
            rating = round(rating * 2) / 2  # Round to 0.5
            ratings_list.append({
                'userId'   : int(user_id),
                'movieId'  : movie['movieId'],
                'rating'   : float(rating),
                'timestamp': np.random.randint(1500000000, 1700000000)
            })

    ratings_df = pd.DataFrame(ratings_list)
    ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'])

    print(f"✅ Indian Movies Dataset Ready!")
    print(f"   → {len(movies_df)} movies (Bollywood + South + Hollywood)")
    print(f"   → {n_users} users")
    print(f"   → {len(ratings_df)} ratings")
    print(f"\n   Industry breakdown:")
    print(movies_df['industry'].value_counts().to_string())

    return movies_df, ratings_df


if __name__ == "__main__":
    movies, ratings = create_indian_movies_dataset()
    movies.to_csv('data/movies.csv', index=False)
    ratings.to_csv('data/ratings.csv', index=False)
    print("\n✅ Saved to data/movies.csv and data/ratings.csv!")
