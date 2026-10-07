"""V8 FastAPI service for a selected, locally available model artifact."""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.inference.runtime import load_predictor


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ticket text cannot be blank")
        return value


class TicketResponse(BaseModel):
    version: str
    label: str
    scores: dict[str, float]
    score_type: str


def create_app(model_path: Path | None = None) -> FastAPI:
    path = model_path or Path(os.environ.get("TICKET_MODEL_PATH", "models/v1.json"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mode = os.environ.get("TICKET_MODE", "demo")
        if mode not in ("demo", "production"):
            raise RuntimeError("TICKET_MODE must be demo or production")
        api_key = os.environ.get("TICKET_API_KEY")
        if not api_key and os.environ.get("TICKET_ALLOW_UNAUTHENTICATED") != "1":
            raise RuntimeError("Set TICKET_API_KEY or TICKET_ALLOW_UNAUTHENTICATED=1 for local use")
        app.state.metadata, app.state.predictor = load_predictor(path)
        if mode == "production":
            if not api_key:
                raise RuntimeError("Production requires TICKET_API_KEY")
            manifest = os.environ.get("TICKET_PROMOTION_MANIFEST")
            if not manifest:
                raise RuntimeError("Production requires TICKET_PROMOTION_MANIFEST")
            from src.inference.promotion import validate_promotion
            validate_promotion(Path(manifest), path, app.state.metadata)
        yield

    app = FastAPI(title="Enterprise Ticket Classifier", version="0.1.0", lifespan=lifespan)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("TICKET_API_KEY")
        if expected and not hmac.compare_digest(x_api_key or "", expected):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": app.state.metadata["version"]}

    @app.post("/predict", response_model=TicketResponse, dependencies=[Depends(require_key)])
    def predict_ticket(request: TicketRequest) -> TicketResponse:
        result = app.state.predictor(request.text)
        version = app.state.metadata["version"]
        score_type = "cosine_similarity" if version in ("v2", "v6") else "model_probability"
        return TicketResponse(version=version, label=result["label"], scores=result["scores"],
                              score_type=score_type)

    return app


app = create_app()
