"""RAG Service using FAISS for historical trading data."""

import json
import os
import pickle
import uuid
from datetime import datetime
from typing import Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

from ..config import settings
from ..models.trading_data import (
    TimeSeriesData,
    MarketAnalysis,
    TradingRecommendation,
)


class RAGService:
    """RAG service for storing and retrieving historical trading context using FAISS."""

    def __init__(self):
        self.persist_directory = settings.chroma_persist_directory
        self._embedding_model = None
        self._index = None
        self._documents = []
        self._metadatas = []
        self._ids = []
        self._dimension = 384  # Default for all-MiniLM-L6-v2

    def _get_embedding_model(self):
        """Get or create the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Loaded embedding model: {settings.embedding_model}")
        return self._embedding_model

    def _get_index(self):
        """Get or create the FAISS index."""
        if self._index is None:
            index_path = os.path.join(self.persist_directory, "faiss.index")
            data_path = os.path.join(self.persist_directory, "data.pkl")

            os.makedirs(self.persist_directory, exist_ok=True)

            if os.path.exists(index_path) and os.path.exists(data_path):
                self._index = faiss.read_index(index_path)
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
                    self._documents = data["documents"]
                    self._metadatas = data["metadatas"]
                    self._ids = data["ids"]
                logger.info(f"Loaded FAISS index with {len(self._documents)} documents")
            else:
                self._index = faiss.IndexFlatL2(self._dimension)
                self._documents = []
                self._metadatas = []
                self._ids = []
                logger.info("Created new FAISS index")

        return self._index

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        model = self._get_embedding_model()
        embedding = model.encode(text)
        return embedding.astype('float32')

    async def add_analysis(
        self,
        analysis: MarketAnalysis,
        recommendation: TradingRecommendation
    ) -> str:
        """Store a market analysis and recommendation in the RAG system."""
        content = self._format_analysis_document(analysis, recommendation)
        doc_id = f"{analysis.symbol}_{analysis.timestamp.isoformat()}_{uuid.uuid4().hex[:8]}"

        metadata = {
            "symbol": analysis.symbol,
            "timestamp": analysis.timestamp.isoformat(),
            "signal": recommendation.signal.value,
            "confidence": recommendation.confidence.value,
            "current_price": analysis.current_price,
            "price_change_24h": analysis.price_change_24h,
            "trend": analysis.trend,
            "document_type": "analysis"
        }

        embedding = self._generate_embedding(content)

        index = self._get_index()
        index.add(np.array([embedding]))
        self._documents.append(content)
        self._metadatas.append(metadata)
        self._ids.append(doc_id)

        logger.info(f"Stored analysis document: {doc_id}")
        return doc_id

    async def add_time_series_pattern(
        self,
        symbol: str,
        pattern_name: str,
        description: str,
        data_points: list[TimeSeriesData],
        outcome: str
    ) -> str:
        """Store a recognized pattern from time series data."""
        content = f"""
Pattern: {pattern_name}
Symbol: {symbol}
Beschreibung: {description}

Datenpunkte:
{self._format_time_series(data_points)}

Ergebnis: {outcome}
"""

        doc_id = f"pattern_{symbol}_{uuid.uuid4().hex[:8]}"

        metadata = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "pattern_name": pattern_name,
            "document_type": "pattern"
        }

        embedding = self._generate_embedding(content)

        index = self._get_index()
        index.add(np.array([embedding]))
        self._documents.append(content)
        self._metadatas.append(metadata)
        self._ids.append(doc_id)

        logger.info(f"Stored pattern document: {doc_id}")
        return doc_id

    async def add_custom_document(
        self,
        content: str,
        document_type: str,
        symbol: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """Add a custom document to the RAG system."""
        doc_id = f"{document_type}_{uuid.uuid4().hex}"

        doc_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "document_type": document_type
        }

        if symbol:
            doc_metadata["symbol"] = symbol

        if metadata:
            doc_metadata.update(metadata)

        embedding = self._generate_embedding(content)

        index = self._get_index()
        index.add(np.array([embedding]))
        self._documents.append(content)
        self._metadatas.append(doc_metadata)
        self._ids.append(doc_id)

        logger.info(f"Stored custom document: {doc_id}")
        return doc_id

    async def query_relevant_context(
        self,
        query: str,
        symbol: Optional[str] = None,
        n_results: int = None,
        document_types: Optional[list[str]] = None
    ) -> list[str]:
        """Query relevant historical context for analysis."""
        if n_results is None:
            n_results = settings.max_context_documents

        index = self._get_index()

        if index.ntotal == 0:
            return []

        query_embedding = self._generate_embedding(query)

        # Search more results to filter
        search_k = min(n_results * 3, index.ntotal)
        distances, indices = index.search(np.array([query_embedding]), search_k)

        documents = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._documents):
                continue

            meta = self._metadatas[idx]

            # Apply filters
            if symbol and meta.get("symbol") != symbol:
                continue
            if document_types and meta.get("document_type") not in document_types:
                continue

            documents.append(self._documents[idx])

            if len(documents) >= n_results:
                break

        logger.info(f"Retrieved {len(documents)} relevant documents for query")
        return documents

    async def get_similar_market_conditions(
        self,
        analysis: MarketAnalysis,
        n_results: int = 5
    ) -> list[str]:
        """Find historical analyses with similar market conditions."""
        query = f"""
Symbol: {analysis.symbol}
Preis: {analysis.current_price}
Trend: {analysis.trend}
Volatilität: {analysis.volatility}
RSI: {analysis.technical_indicators.rsi}
MACD: {analysis.technical_indicators.macd}
24h Änderung: {analysis.price_change_24h}%
"""

        return await self.query_relevant_context(
            query=query,
            symbol=analysis.symbol,
            n_results=n_results,
            document_types=["analysis"]
        )

    def _format_analysis_document(
        self,
        analysis: MarketAnalysis,
        recommendation: TradingRecommendation
    ) -> str:
        """Format analysis and recommendation as document."""
        return f"""
MARKTANALYSE - {analysis.symbol}
Datum: {analysis.timestamp.isoformat()}

Marktdaten:
- Aktueller Preis: {analysis.current_price}
- 24h Änderung: {analysis.price_change_24h}%
- Trend: {analysis.trend}
- Volatilität: {analysis.volatility}

Technische Indikatoren:
- RSI: {analysis.technical_indicators.rsi}
- MACD: {analysis.technical_indicators.macd}
- SMA 20/50/200: {analysis.technical_indicators.sma_20}/{analysis.technical_indicators.sma_50}/{analysis.technical_indicators.sma_200}

Empfehlung: {recommendation.signal.value}
Konfidenz: {recommendation.confidence.value}
Entry: {recommendation.entry_price}
Stop-Loss: {recommendation.stop_loss}
Take-Profit: {recommendation.take_profit}

Begründung: {recommendation.reasoning}

Schlüsselfaktoren: {', '.join(recommendation.key_factors)}

Risiken: {', '.join(recommendation.risks)}
"""

    def _format_time_series(self, data_points: list[TimeSeriesData]) -> str:
        """Format time series data for document."""
        lines = []
        for dp in data_points:
            lines.append(
                f"{dp.timestamp.isoformat()}: O={dp.open} H={dp.high} L={dp.low} C={dp.close} V={dp.volume}"
            )
        return "\n".join(lines)

    async def persist(self):
        """Persist the FAISS index and data to disk."""
        index = self._get_index()

        os.makedirs(self.persist_directory, exist_ok=True)

        index_path = os.path.join(self.persist_directory, "faiss.index")
        data_path = os.path.join(self.persist_directory, "data.pkl")

        faiss.write_index(index, index_path)

        with open(data_path, "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "metadatas": self._metadatas,
                "ids": self._ids
            }, f)

        logger.info("Persisted RAG database to disk")

    async def get_collection_stats(self) -> dict:
        """Get statistics about the RAG collection."""
        index = self._get_index()

        return {
            "collection_name": "trading_history",
            "document_count": len(self._documents),
            "persist_directory": self.persist_directory
        }

    async def delete_documents(
        self,
        symbol: Optional[str] = None,
        before_date: Optional[datetime] = None
    ) -> int:
        """Delete documents from the collection."""
        # FAISS doesn't support direct deletion, so we rebuild the index
        if not symbol:
            return 0

        indices_to_keep = []
        for i, meta in enumerate(self._metadatas):
            if meta.get("symbol") != symbol:
                indices_to_keep.append(i)

        deleted_count = len(self._documents) - len(indices_to_keep)

        if deleted_count > 0:
            # Rebuild index
            new_documents = [self._documents[i] for i in indices_to_keep]
            new_metadatas = [self._metadatas[i] for i in indices_to_keep]
            new_ids = [self._ids[i] for i in indices_to_keep]

            # Create new index
            self._index = faiss.IndexFlatL2(self._dimension)

            if new_documents:
                embeddings = []
                for doc in new_documents:
                    embeddings.append(self._generate_embedding(doc))
                self._index.add(np.array(embeddings))

            self._documents = new_documents
            self._metadatas = new_metadatas
            self._ids = new_ids

            logger.info(f"Deleted {deleted_count} documents")

        return deleted_count
