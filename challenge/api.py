import fastapi
import pandas as pd 
from .model import DelayModel
from fastapi.responses import JSONResponse

app = fastapi.FastAPI()

data = pd.read_csv(filepath_or_buffer="data/data.csv")
model = DelayModel()
features, target = model.preprocess(data=data, target_column="delay")
model.fit(features=features, target=target)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: fastapi.Request) -> dict:
    try:
        payload = await request.json()
        for flight in payload['flights']:
            if not flight["OPERA"] in data["OPERA"].unique():
                return JSONResponse(status_code=400, content={})
            
            if not flight["TIPOVUELO"] in data["TIPOVUELO"].unique():
                return JSONResponse(status_code=400, content={})
            
            if not flight["MES"] in data["MES"].unique():
                return JSONResponse(status_code=400, content={}) 
    
        flights = model.preprocess(pd.DataFrame(payload['flights']))
        return {"predict": model.predict(features=flights)}
    
    except Exception:
        return JSONResponse(status_code=400, content={})
    