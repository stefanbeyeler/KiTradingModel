"""
Standardized Model Loader Base Class
=====================================

Basis-Klasse für Model-Loading mit:
- Timeout-Handling
- GPU/CPU Fallback
- Lazy Loading Support
- Status Reporting
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, TypeVar, Generic
from loguru import logger

T = TypeVar("T")  # Model Type


class ModelLoaderBase(ABC, Generic[T]):
    """
    Abstrakte Basis-Klasse für Model Loader.

    Jeder Service sollte diese Klasse erweitern und die
    abstrakten Methoden implementieren.

    Beispiel:
        class EmbedderModelLoader(ModelLoaderBase[SentenceTransformer]):
            def _load_model_sync(self, path: Path) -> SentenceTransformer:
                return SentenceTransformer(str(path))

            def _create_fresh_model_sync(self) -> SentenceTransformer:
                return SentenceTransformer('all-MiniLM-L6-v2')
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        lazy_load: bool = False,
        load_timeout: float = 120.0,
    ):
        """
        Initialisiert den Model Loader.

        Args:
            model_path: Pfad zum gespeicherten Modell (optional)
            use_gpu: GPU verwenden falls verfügbar
            gpu_device_id: GPU Device ID
            lazy_load: True = Modell erst bei erster Anfrage laden
            load_timeout: Timeout in Sekunden für Model Loading
        """
        self.model_path = Path(model_path) if model_path else None
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.lazy_load = lazy_load
        self.load_timeout = load_timeout

        # State
        self._model: Optional[T] = None
        self._device: Optional[str] = None
        self._is_loaded: bool = False
        self._is_loading: bool = False
        self._load_error: Optional[str] = None
        self._load_lock = asyncio.Lock()

    @property
    def model(self) -> Optional[T]:
        """Gibt das geladene Modell zurück."""
        return self._model

    @property
    def device(self) -> Optional[str]:
        """Gibt das verwendete Device zurück."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """True wenn Modell erfolgreich geladen."""
        return self._is_loaded

    @property
    def is_ready(self) -> bool:
        """True wenn Modell bereit für Anfragen."""
        return self._is_loaded and self._model is not None

    async def load(self) -> None:
        """
        Lädt das Modell (mit Timeout).

        Bei lazy_load=True wird nur vorbereitet, nicht geladen.
        """
        if self.lazy_load:
            logger.info("Lazy loading enabled - model will load on first request")
            return

        await self._do_load()

    async def ensure_loaded(self) -> None:
        """
        Stellt sicher, dass das Modell geladen ist.

        Für Lazy Loading: Lädt bei erster Anfrage.
        """
        if self._is_loaded:
            return

        async with self._load_lock:
            # Double-check nach Lock
            if self._is_loaded:
                return
            await self._do_load()

    async def _do_load(self) -> None:
        """Führt das eigentliche Loading durch."""
        if self._is_loading:
            logger.warning("Model loading already in progress")
            return

        self._is_loading = True
        try:
            # Device Selection
            self._device = self._select_device()
            logger.info(f"Using device: {self._device}")

            # Model laden mit Timeout
            if self.model_path and self.model_path.exists():
                logger.info(f"Loading model from {self.model_path}")
                self._model = await asyncio.wait_for(
                    asyncio.to_thread(self._load_model_sync, self.model_path),
                    timeout=self.load_timeout,
                )
                logger.success(f"Model loaded from {self.model_path}")
            else:
                logger.info("No saved model found, creating fresh model")
                self._model = await asyncio.wait_for(
                    asyncio.to_thread(self._create_fresh_model_sync),
                    timeout=self.load_timeout,
                )
                logger.success("Fresh model created")

            # Model auf Device verschieben
            self._model = self._move_to_device(self._model)

            self._is_loaded = True
            self._load_error = None

        except asyncio.TimeoutError:
            self._load_error = f"Model loading timeout ({self.load_timeout}s)"
            logger.error(self._load_error)
            raise

        except Exception as e:
            self._load_error = str(e)
            logger.error(f"Model loading failed: {e}")
            raise

        finally:
            self._is_loading = False

    def _select_device(self) -> str:
        """Wählt GPU oder CPU mit Fallback."""
        if not self.use_gpu:
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                device = f"cuda:{self.gpu_device_id}"
                gpu_name = torch.cuda.get_device_name(self.gpu_device_id)
                logger.info(f"GPU available: {gpu_name}")
                return device
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")

        if self.use_gpu:
            logger.warning("GPU requested but not available, falling back to CPU")
        return "cpu"

    def _move_to_device(self, model: T) -> T:
        """
        Verschiebt Modell auf das gewählte Device.

        Override für custom Device-Handling.
        """
        if hasattr(model, "to") and self._device:
            return model.to(self._device)
        return model

    @abstractmethod
    def _load_model_sync(self, path: Path) -> T:
        """
        Lädt ein gespeichertes Modell (synchron).

        Muss von Subklassen implementiert werden.

        Args:
            path: Pfad zum gespeicherten Modell

        Returns:
            Geladenes Modell
        """
        pass

    @abstractmethod
    def _create_fresh_model_sync(self) -> T:
        """
        Erstellt ein neues Modell (synchron).

        Muss von Subklassen implementiert werden.

        Returns:
            Neues Modell
        """
        pass

    def save(self, path: Optional[Path] = None) -> None:
        """
        Speichert das Modell.

        Args:
            path: Zielpfad (optional, sonst model_path)
        """
        if not self._model:
            logger.warning("No model to save")
            return

        save_path = path or self.model_path
        if not save_path:
            logger.warning("No save path specified")
            return

        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_model_sync(self._model, save_path)
        logger.info(f"Model saved to {save_path}")

    def _save_model_sync(self, model: T, path: Path) -> None:
        """
        Speichert das Modell (synchron).

        Override für custom Save-Logik.
        """
        import torch
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)

    async def cleanup(self) -> None:
        """Räumt Ressourcen auf."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._device and "cuda" in self._device:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._is_loaded = False
        logger.info("Model resources cleaned up")

    def get_status(self) -> Dict[str, Any]:
        """Gibt Status-Informationen für Health-Check zurück."""
        gpu_available = False
        gpu_name = None

        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return {
            "model_loaded": self._is_loaded,
            "model_ready": self.is_ready,
            "device": self._device,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "load_error": self._load_error,
            "lazy_load": self.lazy_load,
        }
