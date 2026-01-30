import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
DATA_PATH = CURRENT_DIR.parent / "data" / "main_dataset.csv"

df = pd.read_csv(DATA_PATH,  low_memory=False)

def str_to_list(genres):
    return ast.literal_eval(genres)

df["genres"] = df["genres"].apply(str_to_list)


mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df["genres"])
cosine_genres = cosine_similarity(genre_matrix)


tfid = TfidfVectorizer(stop_words="english")
description_matrix = tfid.fit_transform(df["description"])
cosine_decription = cosine_similarity(description_matrix)


weight_genres = 0.5
weight_description = 0.5

cosine_hybride = (
    weight_genres * cosine_genres  +
    weight_description * cosine_decription
)

SAVE_PATH = CURRENT_DIR.parent / "data" / "cosine_hybrid.joblib"

joblib.dump(cosine_hybride, SAVE_PATH)

