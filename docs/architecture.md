# Architecture

This project is organized as a hybrid recommendation platform with separate layers for data preparation, feature generation, ranking, model training, inference, and product delivery.

## High-Level Design

```mermaid
flowchart LR
    DATA[MovieLens + external metadata] --> PREP[Preprocessing layer]
    PREP --> FEATURES[Feature stores]
    FEATURES --> MODELS[Recommendation models]
    MODELS --> BUNDLE[(Versioned model bundle)]
    BUNDLE --> SERVICE[Inference service]
    SERVICE --> API[FastAPI]
    SERVICE --> UI[Streamlit]
```

## Core Components

- `data/`: downloads datasets, normalizes ratings and tags, merges external catalog metadata, and prepares model-ready tables
- `features/`: builds TF-IDF and optional semantic representations for content retrieval
- `ranking/`: adds audience-consensus, audience-language, and query-alignment style signals
- `recommenders/`: contains content, popularity, collaborative, and hybrid ranking logic
- `models/`: trains matrix factorization and autoencoder recommenders
- `services/`: orchestrates training, evaluation, bundle loading, and online inference
- `api/` and `apps/`: expose the recommender through FastAPI and Streamlit

## Training Flow

```mermaid
flowchart TD
    RAW[Raw ratings, tags, metadata] --> NORMALIZE[Normalize and enrich movies]
    NORMALIZE --> CONTENT[Build content features]
    NORMALIZE --> AUDIENCE[Build audience signals]
    CONTENT --> TRAIN[Train recommenders]
    AUDIENCE --> TRAIN
    TRAIN --> TUNE[Tune hybrid weights]
    TUNE --> REPORT[Evaluation report]
    REPORT --> BUNDLE[(Model bundle)]
```

## Online Recommendation Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant API as FastAPI
    participant Service as MovieRecommenderService
    participant Bundle as Model Bundle

    User->>UI: Enter query or select movie
    UI->>API: Recommendation request
    API->>Service: Build request context
    Service->>Bundle: Load models and features
    Service->>Service: Score content, audience, collaborative signals
    Service->>Service: Hybrid combine and rerank
    Service-->>API: Ranked results with explanations
    API-->>UI: Response payload
    UI-->>User: Personalized recommendations
```

## Why This Structure Works

- It keeps offline training concerns separate from the online serving path.
- It allows richer ranking signals, including audience-review language, without coupling everything to one model.
- It supports UI, API, and CLI workflows from the same bundle and service layer.
- It makes the repository easier to explain in interviews, demos, and GitHub documentation.
