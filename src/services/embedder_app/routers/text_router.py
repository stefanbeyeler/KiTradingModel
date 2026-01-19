"""Text embedding endpoints."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models.schemas import (
    TextEmbeddingRequest,
    EmbeddingResponse,
    EmbeddingType,
    SimilarityRequest,
    SimilarityResponse,
    BatchSimilarityRequest,
)
from ..services.embedding_service import embedding_service

router = APIRouter()


@router.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: TextEmbeddingRequest):
    """
    Generate text embeddings.

    - **texts**: List of texts to embed
    - **use_finbert**: Use FinBERT for finance-specific embeddings

    Returns embeddings with dimension info.
    """
    try:
        result = await embedding_service.embed_text(
            texts=request.texts,
            use_finbert=request.use_finbert
        )

        return EmbeddingResponse(
            embeddings=result.embedding.tolist(),
            dimension=result.dimension,
            model=result.model_name,
            embedding_type=result.embedding_type,
            cached=result.cached,
            count=len(request.texts)
        )
    except Exception as e:
        logger.error(f"Error embedding texts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity", response_model=SimilarityResponse)
async def compute_text_similarity(request: SimilarityRequest):
    """
    Compute semantic similarity between two texts.

    - **text1**: First text
    - **text2**: Second text
    - **use_finbert**: Use FinBERT for finance-specific comparison

    Returns cosine similarity score (0-1).
    """
    try:
        result1 = await embedding_service.embed_text(
            texts=[request.text1],
            use_finbert=request.use_finbert
        )
        result2 = await embedding_service.embed_text(
            texts=[request.text2],
            use_finbert=request.use_finbert
        )

        similarity = await embedding_service.compute_similarity(
            result1.embedding[0],
            result2.embedding[0]
        )

        return SimilarityResponse(
            similarity=similarity,
            model=result1.model_name
        )
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-similarity")
async def compute_batch_similarity(request: BatchSimilarityRequest):
    """
    Compute similarity between a query and multiple candidate texts.

    Returns ranked results with similarity scores.

    - **query**: The query text to compare
    - **candidates**: List of candidate texts
    - **use_finbert**: Use FinBERT for finance-specific comparison
    """
    try:
        # Embed query and all candidates together
        all_texts = [request.query] + request.candidates
        result = await embedding_service.embed_text(
            texts=all_texts,
            use_finbert=request.use_finbert
        )

        embeddings = result.embedding
        query_embedding = embeddings[0]

        # Compute similarity between query and each candidate
        results = []
        for i, candidate in enumerate(request.candidates):
            sim = await embedding_service.compute_similarity(
                query_embedding,
                embeddings[i + 1]  # +1 because query is at index 0
            )
            results.append({
                "text": candidate,
                "similarity": round(sim, 4)
            })

        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "query": request.query,
            "results": results,
            "model": result.model_name
        }
    except Exception as e:
        logger.error(f"Error computing batch similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))
