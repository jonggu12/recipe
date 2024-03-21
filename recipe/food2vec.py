from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# 데이터프레임 로드
df = pd.read_csv('recipe/data.csv')  

# Word2Vec 모델 로드
model = Word2Vec.load("recipe/recipe_word2vec_model.model")



# 문서 벡터 리스트 생성 함수
def get_document_vectors(document_list):
    document_embedding_list = []
    for words in document_list:  # 'words' is already a list, no need to strip or split
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
            doc2vec = doc2vec / count  # Averaging the vector
            document_embedding_list.append(doc2vec)
    return np.array(document_embedding_list)

# 주의: 'df['all']' 컬럼을 리스트로 변환해야 합니다.
# 이를 위해, 각 문서를 파싱해야 합니다.
document_vectors = get_document_vectors(df['all'].apply(eval))

# 코사인 유사도 계산
cosine_similarities = cosine_similarity(document_vectors)

class RecipeQuery(BaseModel):
    title: str

@app.post("/recommendations/")
async def get_recommendations(query: RecipeQuery):
    title = query.title
    try:
        recommendations = recommend_function(title)
        return {"recommendations": recommendations}
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
    recommendations = recommend[['recipe_name', 'summary', 'ingredient_name', 'full_step', 'category', 'image_link', 'main_source', 'all']].to_dict('records')
    return recommendations
