from fastapi import FastAPI
import pickle


app = FastAPI()

@app.get('/')
def root():
    return {'hello: world'}

@app.get('/predict')
def predict(
    sepal_length=1,
    sepal_width=1,
    petal_length=1,
    petal_width=1
):
    with open('../models/best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    pred=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]
    return {'flower': float(pred)}

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # 8080 para local, Cloud Run define PORT automaticamente
    uvicorn.run("project_accidents_package.api_file:app", host="0.0.0.0", port=port)
