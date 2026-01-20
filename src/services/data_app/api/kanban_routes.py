"""
Kanban Board API Routes - Backend persistence for Kanban board.

Provides CRUD operations for Kanban board state including:
- Columns (Backlog, In Progress, Done, etc.)
- Cards (Tasks, Bugs, Requirements, Ideas)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger


router = APIRouter(prefix="/kanban", tags=["8. Kanban Board"])


# =============================================================================
# Pydantic Models
# =============================================================================

class KanbanCard(BaseModel):
    """A single Kanban card/task."""
    id: str
    title: str
    description: Optional[str] = ""
    columnId: str
    classification: str = "task"  # task, bug, requirement, idee
    service: Optional[str] = None
    priority: str = "medium"  # low, medium, high
    createdAt: str
    updatedAt: Optional[str] = None


class KanbanColumn(BaseModel):
    """A Kanban column."""
    id: str
    name: str
    color: Optional[str] = "#64c8ff"


class KanbanState(BaseModel):
    """Complete Kanban board state."""
    columns: List[KanbanColumn]
    cards: List[KanbanCard]
    lastModified: Optional[str] = None


class CardCreate(BaseModel):
    """Request model for creating a card."""
    title: str
    description: Optional[str] = ""
    columnId: str
    classification: str = "task"
    service: Optional[str] = None
    priority: str = "medium"


class CardUpdate(BaseModel):
    """Request model for updating a card."""
    title: Optional[str] = None
    description: Optional[str] = None
    columnId: Optional[str] = None
    classification: Optional[str] = None
    service: Optional[str] = None
    priority: Optional[str] = None


class ColumnCreate(BaseModel):
    """Request model for creating a column."""
    name: str
    color: Optional[str] = "#64c8ff"


class ColumnUpdate(BaseModel):
    """Request model for updating a column."""
    name: Optional[str] = None
    color: Optional[str] = None


# =============================================================================
# Storage Service
# =============================================================================

class KanbanStorageService:
    """Service for persisting Kanban board state."""

    DEFAULT_COLUMNS = [
        {"id": "backlog", "name": "Backlog", "color": "#6b7280"},
        {"id": "todo", "name": "To Do", "color": "#3b82f6"},
        {"id": "in-progress", "name": "In Progress", "color": "#f59e0b"},
        {"id": "review", "name": "Review", "color": "#8b5cf6"},
        {"id": "done", "name": "Done", "color": "#10b981"},
    ]

    def __init__(self, storage_path: str = None):
        """Initialize storage service."""
        if storage_path is None:
            storage_path = os.getenv("KANBAN_STORAGE_PATH", "/app/data/kanban/kanban_state.json")
        self._storage_path = Path(storage_path)
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Optional[Dict] = None
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk."""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
                logger.info(f"Loaded Kanban state: {len(self._state.get('cards', []))} cards")
            except Exception as e:
                logger.error(f"Error loading Kanban state: {e}")
                self._initialize_default_state()
        else:
            self._initialize_default_state()

    def _initialize_default_state(self) -> None:
        """Initialize with default state."""
        self._state = {
            "columns": self.DEFAULT_COLUMNS,
            "cards": [],
            "lastModified": datetime.now(timezone.utc).isoformat()
        }
        self._save_state()
        logger.info("Initialized default Kanban state")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self._state["lastModified"] = datetime.now(timezone.utc).isoformat()
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving Kanban state: {e}")
            raise

    def get_state(self) -> Dict:
        """Get complete board state."""
        return self._state.copy()

    def set_state(self, state: Dict) -> Dict:
        """Set complete board state."""
        self._state = state
        self._save_state()
        return self._state.copy()

    # Column operations
    def get_columns(self) -> List[Dict]:
        """Get all columns."""
        return self._state.get("columns", [])

    def add_column(self, column: Dict) -> Dict:
        """Add a new column."""
        columns = self._state.get("columns", [])
        columns.append(column)
        self._state["columns"] = columns
        self._save_state()
        return column

    def update_column(self, column_id: str, updates: Dict) -> Optional[Dict]:
        """Update a column."""
        columns = self._state.get("columns", [])
        for i, col in enumerate(columns):
            if col["id"] == column_id:
                columns[i] = {**col, **updates}
                self._state["columns"] = columns
                self._save_state()
                return columns[i]
        return None

    def delete_column(self, column_id: str) -> bool:
        """Delete a column and its cards."""
        columns = self._state.get("columns", [])
        cards = self._state.get("cards", [])

        # Remove column
        self._state["columns"] = [c for c in columns if c["id"] != column_id]

        # Remove cards in this column
        self._state["cards"] = [c for c in cards if c["columnId"] != column_id]

        self._save_state()
        return True

    # Card operations
    def get_cards(self, column_id: Optional[str] = None) -> List[Dict]:
        """Get cards, optionally filtered by column."""
        cards = self._state.get("cards", [])
        if column_id:
            cards = [c for c in cards if c["columnId"] == column_id]
        return cards

    def get_card(self, card_id: str) -> Optional[Dict]:
        """Get a single card by ID."""
        for card in self._state.get("cards", []):
            if card["id"] == card_id:
                return card
        return None

    def add_card(self, card: Dict) -> Dict:
        """Add a new card."""
        cards = self._state.get("cards", [])
        cards.append(card)
        self._state["cards"] = cards
        self._save_state()
        return card

    def update_card(self, card_id: str, updates: Dict) -> Optional[Dict]:
        """Update a card."""
        cards = self._state.get("cards", [])
        for i, card in enumerate(cards):
            if card["id"] == card_id:
                cards[i] = {**card, **updates, "updatedAt": datetime.now(timezone.utc).isoformat()}
                self._state["cards"] = cards
                self._save_state()
                return cards[i]
        return None

    def delete_card(self, card_id: str) -> bool:
        """Delete a card."""
        cards = self._state.get("cards", [])
        self._state["cards"] = [c for c in cards if c["id"] != card_id]
        self._save_state()
        return True

    def move_card(self, card_id: str, target_column_id: str) -> Optional[Dict]:
        """Move a card to a different column."""
        return self.update_card(card_id, {"columnId": target_column_id})


# Singleton instance
kanban_storage = KanbanStorageService()


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/state", response_model=KanbanState, summary="Get complete board state")
async def get_board_state():
    """
    Get the complete Kanban board state including all columns and cards.
    """
    return kanban_storage.get_state()


@router.put("/state", response_model=KanbanState, summary="Set complete board state")
async def set_board_state(state: KanbanState):
    """
    Replace the complete Kanban board state.
    Used for bulk updates or imports.
    """
    return kanban_storage.set_state(state.model_dump())


# Column endpoints
@router.get("/columns", response_model=List[KanbanColumn], summary="Get all columns")
async def get_columns():
    """Get all Kanban columns."""
    return kanban_storage.get_columns()


@router.post("/columns", response_model=KanbanColumn, summary="Create a column")
async def create_column(column: ColumnCreate):
    """Create a new Kanban column."""
    import uuid
    new_column = {
        "id": str(uuid.uuid4())[:8],
        "name": column.name,
        "color": column.color or "#64c8ff"
    }
    return kanban_storage.add_column(new_column)


@router.put("/columns/{column_id}", response_model=KanbanColumn, summary="Update a column")
async def update_column(column_id: str, column: ColumnUpdate):
    """Update a Kanban column."""
    updates = column.model_dump(exclude_none=True)
    result = kanban_storage.update_column(column_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Column {column_id} not found")
    return result


@router.delete("/columns/{column_id}", summary="Delete a column")
async def delete_column(column_id: str):
    """Delete a Kanban column and all its cards."""
    kanban_storage.delete_column(column_id)
    return {"status": "ok", "message": f"Column {column_id} deleted"}


# Card endpoints
@router.get("/cards", response_model=List[KanbanCard], summary="Get all cards")
async def get_cards(column_id: Optional[str] = None):
    """Get all Kanban cards, optionally filtered by column."""
    return kanban_storage.get_cards(column_id)


@router.get("/cards/{card_id}", response_model=KanbanCard, summary="Get a card")
async def get_card(card_id: str):
    """Get a single Kanban card by ID."""
    card = kanban_storage.get_card(card_id)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Card {card_id} not found")
    return card


@router.post("/cards", response_model=KanbanCard, summary="Create a card")
async def create_card(card: CardCreate):
    """Create a new Kanban card."""
    import uuid
    new_card = {
        "id": str(uuid.uuid4())[:8],
        "title": card.title,
        "description": card.description or "",
        "columnId": card.columnId,
        "classification": card.classification,
        "service": card.service,
        "priority": card.priority,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": None
    }
    return kanban_storage.add_card(new_card)


@router.put("/cards/{card_id}", response_model=KanbanCard, summary="Update a card")
async def update_card(card_id: str, card: CardUpdate):
    """Update a Kanban card."""
    updates = card.model_dump(exclude_none=True)
    result = kanban_storage.update_card(card_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Card {card_id} not found")
    return result


@router.delete("/cards/{card_id}", summary="Delete a card")
async def delete_card(card_id: str):
    """Delete a Kanban card."""
    kanban_storage.delete_card(card_id)
    return {"status": "ok", "message": f"Card {card_id} deleted"}


@router.put("/cards/{card_id}/move", response_model=KanbanCard, summary="Move a card")
async def move_card(card_id: str, target_column_id: str):
    """Move a Kanban card to a different column."""
    result = kanban_storage.move_card(card_id, target_column_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Card {card_id} not found")
    return result
