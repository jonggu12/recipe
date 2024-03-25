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
import firebase_admin
from firebase_admin import credentials, db
from typing import Dict
import operator

cred = credentials.Certificate('senior-752be-firebase-adminsdk-k8csi-2b1357c54a.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://senior-752be-default-rtdb.firebaseio.com/'
})





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
df = pd.read_csv('data.csv', encoding='utf-8')



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





@app.get("/most_clicked_recipe")
async def get_most_clicked_recipe():
    ref = db.reference('user_clicks')
    clicks = ref.get()
    if not clicks:
        raise HTTPException(status_code=404, detail="No clicks data found")

    # 레시피 클릭 수 집계
    recipe_clicks: Dict[str, int] = {}
    for click_id, click_info in clicks.items():
        recipe_name = click_info['recipe_name']
        if recipe_name in recipe_clicks:
            recipe_clicks[recipe_name] += 1
        else:
            recipe_clicks[recipe_name] = 1

    # 가장 많이 클릭된 레시피 찾기
    most_clicked_recipe = max(recipe_clicks.items(), key=operator.itemgetter(1))[0]
    return {"most_clicked_recipe": most_clicked_recipe}









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

@app.get("/recommendations/most_clicked")
async def get_most_clicked_recommendations():
    try:
        # 가장 많이 클릭된 레시피 이름 조회
        most_clicked_recipe = await get_most_clicked_recipe()
        most_clicked_recipe_name = most_clicked_recipe["most_clicked_recipe"]
        
        # 유사한 레시피 추천
        recommendations = recommend_function(most_clicked_recipe_name)
        content = jsonable_encoder({"recommendations": recommendations})
        return JSONResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

def recommend_function(most_clicked_recipe, num_recommendations=5):
    

    # cleaned_recipe를 사용하여 데이터셋 내 레시피 존재 여부 확인
    if most_clicked_recipe not in df['recipe_name'].values:
        raise ValueError("Recipe not found in the dataset.")
    
    idx = df.index[df['recipe_name'] == most_clicked_recipe].tolist()[0]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    

    sim_scores = sim_scores[0:]
    
    # 선택된 레시피들의 인덱스
    recipe_indices = [i[0] for i in sim_scores]
    
    # 선택된 레시피들의 정보
    recommendations = df.iloc[recipe_indices]
    recommendations = recommendations[['recipe_name', 'summary', 'ingredient_name', 'full_step', 'category', 'recipe_image_link']].to_dict('records')
    
    # 최대 num_recommendations 개수까지 레시피 추천
    if len(recommendations) < num_recommendations:
        additional_recipes = df.sample(n=num_recommendations-len(recommendations)).to_dict('records')
        recommendations.extend(additional_recipes)

    return recommendations[:num_recommendations]
    









    
    
    
# 랜덤 레시피 추천
    

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
