import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from pipeline.preprocessor import Preprocessor
from pipeline.runner import PipelineRunner
from pipeline.xai import XAILayer
from agent.agent import ForgeryAgent
from routers import analyze


preprocessor = Preprocessor()
runner = PipelineRunner()
agent = ForgeryAgent()
xai = XAILayer()

analyze.set_dependencies(preprocessor, runner, agent, xai)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("uploads", exist_ok=True)
    print("Models loaded:", runner.get_model_status())
    yield


app = FastAPI(title="Document Forgery Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(analyze.router)


@app.get("/")
async def root():
    return {"message": "Document Forgery Detection API", "version": "1.0.0"}