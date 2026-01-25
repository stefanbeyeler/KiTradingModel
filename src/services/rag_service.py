"""RAG Service using FAISS for historical trading data."""

import hashlib
import json
import os
import pickle
import uuid
from datetime import datetime
from typing import Optional
import numpy as np
import faiss
import torch
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
        self.persist_directory = settings.faiss_persist_directory
        self._embedding_model = None
        self._index = None
        self._gpu_index = None
        self._gpu_resources = None
        self._documents = []
        self._metadatas = []
        self._ids = []
        self._content_hashes: set[str] = set()  # For duplicate detection
        self._dimension = 384  # Default for all-MiniLM-L6-v2
        self._device = settings.device
        self._use_gpu_faiss = settings.faiss_use_gpu and self._check_faiss_gpu()

        logger.info(f"RAG Service initialized - Device: {self._device}, FAISS GPU: {self._use_gpu_faiss}")

    def _check_faiss_gpu(self) -> bool:
        """Check if FAISS GPU is available."""
        try:
            return hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available()
        except Exception:
            return False

    def _compute_content_hash(self, content: str) -> str:
        """Compute a hash for content to detect duplicates."""
        # Normalize content: strip whitespace and convert to lowercase for comparison
        normalized = content.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:32]

    def is_duplicate(self, content: str) -> bool:
        """Check if content already exists in the RAG database."""
        content_hash = self._compute_content_hash(content)
        return content_hash in self._content_hashes

    def _get_embedding_model(self):
        """Get or create the embedding model with GPU support."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                settings.embedding_model,
                device=self._device
            )

            # Use half precision on GPU to save VRAM
            if self._device == "cuda" and settings.use_half_precision:
                self._embedding_model = self._embedding_model.half()
                logger.info("Using FP16 precision for embeddings")

            logger.info(f"Loaded embedding model: {settings.embedding_model} on {self._device}")
        return self._embedding_model

    def _get_index(self):
        """Get or create the FAISS index with optional GPU support."""
        if self._index is None:
            index_path = os.path.join(self.persist_directory, "faiss.index")
            data_path = os.path.join(self.persist_directory, "data.pkl")
            index_bak_path = index_path + ".bak"
            data_bak_path = data_path + ".bak"

            os.makedirs(self.persist_directory, exist_ok=True)

            loaded = False

            # Try loading from primary files
            if os.path.exists(index_path) and os.path.exists(data_path):
                loaded = self._try_load_index(index_path, data_path, "primary")

            # If primary load failed, try backup files
            if not loaded and os.path.exists(index_bak_path) and os.path.exists(data_bak_path):
                logger.warning("Primary files corrupt, attempting to restore from backup...")
                loaded = self._try_load_index(index_bak_path, data_bak_path, "backup")

                if loaded:
                    # Restore backup as primary
                    try:
                        import shutil
                        shutil.copy2(data_bak_path, data_path)
                        shutil.copy2(index_bak_path, index_path)
                        logger.info("Restored backup files as primary")
                    except Exception as e:
                        logger.warning(f"Could not restore backup as primary: {e}")

            # If all loading attempts failed, create new index
            if not loaded:
                self._index = faiss.IndexFlatL2(self._dimension)
                self._documents = []
                self._metadatas = []
                self._ids = []
                self._content_hashes = set()
                logger.info("Created new FAISS index")

            # Move index to GPU if available and enabled
            if self._use_gpu_faiss:
                self._index = self._move_index_to_gpu(self._index)

        return self._index

    def _try_load_index(self, index_path: str, data_path: str, source: str) -> bool:
        """Try to load index from given paths. Returns True on success."""
        try:
            self._index = faiss.read_index(index_path)
            with open(data_path, "rb") as f:
                data = pickle.load(f)

                if not isinstance(data, dict):
                    raise ValueError("Invalid data format: not a dictionary")

                self._documents = data.get("documents", [])
                self._metadatas = data.get("metadatas", [])
                self._ids = data.get("ids", [])

                # Load content hashes if available, otherwise rebuild from documents
                if "content_hashes" in data:
                    self._content_hashes = data["content_hashes"]
                else:
                    # Rebuild hashes from existing documents
                    self._content_hashes = {
                        self._compute_content_hash(doc) for doc in self._documents
                    }
                    logger.info(f"Rebuilt {len(self._content_hashes)} content hashes")

            logger.info(f"Loaded FAISS index from {source} with {len(self._documents)} documents")
            return True

        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            logger.error(f"Failed to load {source} index files (corrupt): {e}")
            self._index = None
            self._documents = []
            self._metadatas = []
            self._ids = []
            self._content_hashes = set()
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading {source} index: {e}")
            self._index = None
            self._documents = []
            self._metadatas = []
            self._ids = []
            self._content_hashes = set()
            return False

    def _move_index_to_gpu(self, index):
        """Move FAISS index to GPU for faster search."""
        try:
            if self._gpu_resources is None:
                self._gpu_resources = faiss.StandardGpuResources()
                # Limit GPU memory usage for FAISS (reserve memory for embeddings)
                self._gpu_resources.setTempMemory(256 * 1024 * 1024)  # 256MB

            gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
            logger.info("FAISS index moved to GPU")
            return gpu_index
        except Exception as e:
            logger.warning(f"Failed to move FAISS index to GPU: {e}. Using CPU.")
            self._use_gpu_faiss = False
            return index

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using GPU if available."""
        model = self._get_embedding_model()
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype('float32')

    def _generate_embeddings_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts in batch (more efficient on GPU)."""
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.astype('float32')

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
        metadata: Optional[dict] = None,
        skip_duplicates: bool = True
    ) -> Optional[str]:
        """Add a custom document to the RAG system.

        Args:
            content: Document content
            document_type: Type of document
            symbol: Trading symbol (optional)
            metadata: Additional metadata (optional)
            skip_duplicates: If True, skip documents that already exist

        Returns:
            Document ID if stored, None if skipped as duplicate
        """
        # Check for duplicates
        content_hash = self._compute_content_hash(content)
        if skip_duplicates and content_hash in self._content_hashes:
            logger.debug(f"Skipping duplicate document (type: {document_type}, symbol: {symbol})")
            return None

        doc_id = f"{document_type}_{uuid.uuid4().hex}"

        doc_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "document_type": document_type,
            "content_hash": content_hash
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
        self._content_hashes.add(content_hash)

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

        # Search more results when filtering by symbol (semantic search may return other symbols first)
        # XAUUSD can appear at position 500+ in semantic search, so we need large search_k
        if symbol:
            search_k = min(max(n_results * 500, 2500), index.ntotal)
        else:
            search_k = min(n_results * 10, index.ntotal)
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

    async def query_relevant_context_with_details(
        self,
        query: str,
        symbol: Optional[str] = None,
        n_results: int = None,
        document_types: Optional[list[str]] = None
    ) -> tuple[list[str], dict]:
        """
        Query relevant historical context with detailed logging information.
        Returns: (documents, context_details_dict)
        """
        import time

        if n_results is None:
            n_results = settings.max_context_documents

        index = self._get_index()

        context_details = {
            "query_text": query,
            "documents_retrieved": 0,
            "documents_used": 0,
            "document_details": [],
            "filter_symbol": symbol,
            "filter_document_types": document_types or [],
            "embedding_model": settings.embedding_model,
            "embedding_dimension": self._dimension,
            "search_k": 0,
            "embedding_time_ms": 0,
            "search_time_ms": 0,
        }

        if index.ntotal == 0:
            return [], context_details

        # Measure embedding time
        embed_start = time.time()
        query_embedding = self._generate_embedding(query)
        context_details["embedding_time_ms"] = (time.time() - embed_start) * 1000

        # Search more results when filtering by symbol (semantic search may return other symbols first)
        # XAUUSD can appear at position 500+ in semantic search, so we need large search_k
        if symbol:
            search_k = min(max(n_results * 500, 2500), index.ntotal)
        else:
            search_k = min(n_results * 10, index.ntotal)
        context_details["search_k"] = search_k

        # Measure search time
        search_start = time.time()
        distances, indices = index.search(np.array([query_embedding]), search_k)
        context_details["search_time_ms"] = (time.time() - search_start) * 1000

        context_details["documents_retrieved"] = len([i for i in indices[0] if i >= 0])

        documents = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue

            meta = self._metadatas[idx]
            doc_id = self._ids[idx] if idx < len(self._ids) else f"doc_{idx}"
            content = self._documents[idx]

            # Calculate similarity score (convert L2 distance to similarity)
            distance = float(distances[0][i]) if i < len(distances[0]) else 0
            similarity_score = 1 / (1 + distance)  # Convert distance to similarity

            # Apply filters
            if symbol and meta.get("symbol") != symbol:
                continue
            if document_types and meta.get("document_type") not in document_types:
                continue

            # Create document detail entry
            doc_detail = {
                "id": doc_id,
                "type": meta.get("document_type", "unknown"),
                "symbol": meta.get("symbol"),
                "timestamp": meta.get("timestamp"),
                "similarity_score": round(similarity_score, 4),
                "l2_distance": round(distance, 4),
                "content_preview": content[:500] + "..." if len(content) > 500 else content,
                "content_length": len(content),
                "metadata": {k: v for k, v in meta.items() if k not in ["symbol", "timestamp", "document_type"]},
            }
            context_details["document_details"].append(doc_detail)

            documents.append(content)

            if len(documents) >= n_results:
                break

        context_details["documents_used"] = len(documents)

        logger.info(
            f"RAG Query: Retrieved {context_details['documents_retrieved']} docs, "
            f"used {context_details['documents_used']} after filtering "
            f"(embed: {context_details['embedding_time_ms']:.1f}ms, "
            f"search: {context_details['search_time_ms']:.1f}ms)"
        )

        return documents, context_details

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
        """Persist the FAISS index and data to disk with atomic writes and backup."""
        index = self._get_index()

        os.makedirs(self.persist_directory, exist_ok=True)

        index_path = os.path.join(self.persist_directory, "faiss.index")
        data_path = os.path.join(self.persist_directory, "data.pkl")
        index_tmp_path = index_path + ".tmp"
        data_tmp_path = data_path + ".tmp"
        index_bak_path = index_path + ".bak"
        data_bak_path = data_path + ".bak"

        try:
            # Step 1: Write to temporary files first
            if self._use_gpu_faiss:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, index_tmp_path)
            else:
                faiss.write_index(index, index_tmp_path)

            with open(data_tmp_path, "wb") as f:
                pickle.dump({
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "ids": self._ids,
                    "content_hashes": self._content_hashes
                }, f)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Step 2: Verify the temp files are valid before replacing
            self._verify_persisted_data(data_tmp_path, index_tmp_path)

            # Step 3: Create backup of existing files (if they exist)
            if os.path.exists(data_path):
                if os.path.exists(data_bak_path):
                    os.remove(data_bak_path)
                os.rename(data_path, data_bak_path)

            if os.path.exists(index_path):
                if os.path.exists(index_bak_path):
                    os.remove(index_bak_path)
                os.rename(index_path, index_bak_path)

            # Step 4: Atomic rename of temp files to final files
            os.rename(data_tmp_path, data_path)
            os.rename(index_tmp_path, index_path)

            logger.info(f"Persisted RAG database to disk ({len(self._documents)} docs, {len(self._content_hashes)} unique hashes)")

        except Exception as e:
            # Cleanup temp files on failure
            for tmp_path in [data_tmp_path, index_tmp_path]:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            logger.error(f"Failed to persist RAG database: {e}")
            raise

    def _verify_persisted_data(self, data_path: str, index_path: str):
        """Verify that persisted files are valid and can be loaded."""
        # Verify pickle file
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            if not isinstance(data, dict):
                raise ValueError("Invalid data.pkl format: not a dictionary")
            required_keys = {"documents", "metadatas", "ids"}
            if not required_keys.issubset(data.keys()):
                raise ValueError(f"Invalid data.pkl format: missing keys {required_keys - data.keys()}")

        # Verify FAISS index
        test_index = faiss.read_index(index_path)
        if test_index is None:
            raise ValueError("Invalid faiss.index: could not load")

    async def get_collection_stats(self) -> dict:
        """Get statistics about the RAG collection."""
        index = self._get_index()

        return {
            "collection_name": "trading_history",
            "document_count": len(self._documents),
            "device": self._device,
            "faiss_gpu": self._use_gpu_faiss,
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
