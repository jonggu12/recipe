from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
import os
from dotenv import load_dotenv
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sklearn.metrics.pairwise import cosine_similarity
import random

# 환경변수 로드
load_dotenv()

# Weaviate 클라이언트 설정
WEAVIATE_CLUSTER_URL = os.getenv('WEAVIATE_CLUSTER_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

client = weaviate.Client(
    url=WEAVIATE_CLUSTER_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY, "X-Cohere-Api-Key": COHERE_API_KEY})

app = FastAPI()

# 데이터프레임 로드
df = pd.read_csv('data.csv')  

# Word2Vec 모델 로드
model = Word2Vec.load("recipe_word2vec_model.model")


class RecipeQuery(BaseModel):
    query: str = None  # 기본 쿼리 (옵션)
    health_goal: str = None  # 건강 목적 (옵션)
    ingredients: list[str] = []  # 재료 목록 (옵션)


@app.post("/recipes/")
async def get_recipes(recipe_query: RecipeQuery):
    try:
        # where 절 구성을 위한 조건들을 담을 리스트 초기화
        conditions = []
        
        # 건강 목적이 제공된 경우 conditions 리스트에 추가
        if recipe_query.health_goal:
            conditions.append({
                "path": ["category"],
                "operator": "Equal",
                "valueString": recipe_query.health_goal
            })
        
        # 재료가 제공된 경우, 각 재료에 대해 conditions 리스트에 추가
        for ingredient in recipe_query.ingredients:
            conditions.append({
                "path": ["ingredient_name"],
                "operator": "Equal",
                "valueString": ingredient
            })
        
        # 모든 조건을 And 연산자로 결합
        if conditions:
            where_clause = {
                "operator": "And",
                "operands": conditions
            }
        else:
            where_clause = {}
        
        # Weaviate 쿼리 실행
        response = client.query.get("Recipe", [
            "recipe_name",
            "summary",
            "ingredient_name",
            "full_step",
            "category",
            "image_link",
        ]).with_where(where_clause).with_limit(10).do()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# 문서 벡터 리스트 생성 함수
def get_document_vectors(document_list):
    document_embedding_list = []
    for words in document_list:
        doc2vec = None
        count = 0
        for word in words:
            if word in model.wv.index_to_key:
                count += 1
                if doc2vec is None:
                    doc2vec = model.wv[word]
                else:
                    doc2vec = doc2vec + model.wv[word]
        if doc2vec is not None:
            doc2vec = doc2vec / count  # 벡터 평균 계산
            document_embedding_list.append(doc2vec)
    return np.array(document_embedding_list)

document_vectors = get_document_vectors(df['all'].apply(eval))

def calculate_cosine_similarity(document_vectors):
    return cosine_similarity(document_vectors)

cosine_similarities = calculate_cosine_similarity(document_vectors)

@app.get("/recommendations/{title}")
async def get_recommendations(title: str):
    try:
        recommendations = recommend_function(title)
        content = jsonable_encoder({"recommendations": recommendations})
        return JSONResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

def recommend_function(title):
    indices = pd.Series(df.index, index=df['recipe_name']).drop_duplicates()
    if title not in indices:
        raise ValueError("Recipe not found.")
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    recipe_indices = [i[0] for i in sim_scores]
    recommend = df.iloc[recipe_indices]
    recommendations = recommend[['recipe_name', 'summary', 'ingredient_name', 'full_step', 'category', 'recipe_image_link']].to_dict('records')
    return recommendations







class RecipeResponse(BaseModel):
    recipe_name: str
    summary: str
    ingredient_name: str
    full_step: str
    category: str
    recipe_image_link: str

@app.post("/random_recipes/")
async def get_random_recipes():
    try:
        random_indices = random.sample(range(len(df)), 5)  # 데이터프레임에서 무작위 인덱스 선택
        random_recipes = df.iloc[random_indices]

        response_data = []
        for _, row in random_recipes.iterrows():
            recipe = RecipeResponse(
                recipe_name=row['recipe_name'],
                summary=row['summary'],
                ingredient_name=row['ingredient_name'],
                full_step=row['full_step'],
                category=row['category'],
                recipe_image_link=row['recipe_image_link']
            )
            response_data.append(recipe)

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
