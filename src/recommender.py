import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
DATA_PATH = CURRENT_DIR.parent / "data" / "main_dataset.csv"

df = pd.read_csv(DATA_PATH,  low_memory=False)


def safe_literal_eval(val):
    if pd.isna(val) or val == "[]":
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

df["genres"] = df["genres"].apply(safe_literal_eval)

cosine_sim_hybride = joblib.load("data/cosine_hybrid.joblib")

def recommend_by_hybride(title_query, n = 5):

    titles = [i.strip() for i in title_query.split(",") if i.strip()]

    if not titles:
        print("Please write at least one movie title.")
        return None

    indexes = []
    for title in titles:
        matches = df[df["title"].str.lower() == title.lower()]

        if matches.empty:
            return f"Sorry, we don't have: '{title}'"
    
        indexes.append(matches.index[0])

    print(indexes)

    sim_vectors = [cosine_sim_hybride[idx] for idx in indexes]

    avg_sim = np.mean(sim_vectors, axis = 0)
    avg_year = df.loc[indexes, "year"].mean()

    for idx in indexes:
        avg_sim[idx] = -1

    sim_scores = list(enumerate(avg_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:n+5]
    movie_indices = [i[0] for i in sim_scores]
    
    result = df.iloc[movie_indices][["title", "year", "rating", "genres"]].reset_index(drop=True)

    result["diff_years"] = abs(avg_year - result["year"])

    result = result.sort_values(by = ["rating", "diff_years"], ascending=[False, True]).reset_index(drop=True)
    
    result = result.drop(columns=["diff_years"])

    return result[0:n].to_dict(orient="records")

# print(recommend_by_hybride("The Matrix, KPop Demon Hunters"))





