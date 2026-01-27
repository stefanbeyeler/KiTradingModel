"""MT5 Trade Service.

HTTP-Client zum Data Service für Trade- und Terminal-Daten.
Bietet Caching und Transformationen für das Workplace Frontend.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from loguru import logger

from ..config import settings
from ..models.schemas import (
    MT5Terminal,
    MT5TerminalStatus,
    MT5Trade,
    MT5TradeSetupLink,
    MT5TradeWithSetup,
    MT5PerformanceMetrics,
    MT5TradeStatus,
    MT5TradeType,
)


class MT5TradeService:
    """Service für MT5 Trade- und Terminal-Daten vom Data Service."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = f"{settings.data_service_url}/api/v1/mt5"

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialisierter HTTP-Client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=settings.http_timeout_seconds,
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Schliesst den HTTP-Client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # =========================================================================
    # Terminal-Operationen
    # =========================================================================

    async def get_terminals(self, active_only: bool = True) -> list[MT5Terminal]:
        """Holt alle Terminals vom Data Service."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}/terminals",
                params={"active_only": active_only, "limit": 100},
            )

            if response.status_code != 200:
                logger.warning(f"Failed to fetch terminals: {response.status_code}")
                return []

            data = response.json()
            terminals = []
            for t in data.get("terminals", []):
                terminals.append(self._parse_terminal(t))
            return terminals

        except Exception as e:
            logger.error(f"Error fetching terminals: {e}")
            return []

    async def get_terminal(self, terminal_id: str) -> Optional[MT5Terminal]:
        """Holt ein einzelnes Terminal."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/terminals/{terminal_id}")

            if response.status_code != 200:
                return None

            return self._parse_terminal(response.json())

        except Exception as e:
            logger.error(f"Error fetching terminal {terminal_id}: {e}")
            return None

    def _parse_terminal(self, data: dict) -> MT5Terminal:
        """Parsed Terminal-Daten und berechnet Status."""
        status = MT5TerminalStatus.UNKNOWN

        if data.get("last_heartbeat"):
            last_hb = data["last_heartbeat"]
            if isinstance(last_hb, str):
                last_hb = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - last_hb < timedelta(minutes=5):
                status = MT5TerminalStatus.ONLINE
            else:
                status = MT5TerminalStatus.OFFLINE

        return MT5Terminal(
            terminal_id=data["terminal_id"],
            name=data["name"],
            account_number=data["account_number"],
            broker_name=data.get("broker_name"),
            server=data.get("server"),
            account_type=data.get("account_type", "real"),
            currency=data.get("currency", "USD"),
            leverage=data.get("leverage"),
            is_active=data.get("is_active", True),
            last_heartbeat=data.get("last_heartbeat"),
            status=status,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    # =========================================================================
    # Trade-Operationen
    # =========================================================================

    async def get_trades(
        self,
        terminal_id: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
        include_links: bool = True,
    ) -> tuple[list[MT5TradeWithSetup], int]:
        """
        Holt Trades mit optionalen Filtern.

        Returns:
            Tuple aus (Trade-Liste, Gesamtanzahl)
        """
        try:
            client = await self._get_client()
            params = {
                "limit": limit,
                "offset": offset,
                "include_stats": False,
            }

            if terminal_id:
                params["terminal_id"] = terminal_id
            if symbol:
                params["symbol"] = symbol.upper()
            if status:
                params["status"] = status
            if since:
                params["since"] = since.isoformat()
            if until:
                params["until"] = until.isoformat()

            response = await client.get(f"{self._base_url}/trades", params=params)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch trades: {response.status_code}")
                return [], 0

            data = response.json()
            trades = []

            # Terminal-Map für Namen
            terminals = await self.get_terminals()
            terminal_map = {t.terminal_id: t.name for t in terminals}

            for t in data.get("trades", []):
                trade = self._parse_trade(t)
                trade_with_setup = MT5TradeWithSetup(
                    **trade.model_dump(),
                    terminal_name=terminal_map.get(trade.terminal_id),
                )

                # Setup-Link holen wenn gewünscht
                if include_links:
                    link = await self.get_trade_link(trade.trade_id)
                    trade_with_setup.setup = link

                trades.append(trade_with_setup)

            return trades, data.get("total", len(trades))

        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return [], 0

    async def get_trade(self, trade_id: str, include_link: bool = True) -> Optional[MT5TradeWithSetup]:
        """Holt einen einzelnen Trade mit optionalem Setup-Link."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/trades/{trade_id}")

            if response.status_code != 200:
                return None

            trade = self._parse_trade(response.json())

            # Terminal-Name holen
            terminal = await self.get_terminal(trade.terminal_id)
            terminal_name = terminal.name if terminal else None

            trade_with_setup = MT5TradeWithSetup(
                **trade.model_dump(),
                terminal_name=terminal_name,
            )

            # Setup-Link holen
            if include_link:
                link = await self.get_trade_link(trade_id)
                trade_with_setup.setup = link

            return trade_with_setup

        except Exception as e:
            logger.error(f"Error fetching trade {trade_id}: {e}")
            return None

    async def get_recent_trades(self, limit: int = 10) -> list[MT5TradeWithSetup]:
        """Holt die letzten Trades."""
        trades, _ = await self.get_trades(limit=limit, include_links=False)
        return trades

    async def get_open_trades(self) -> list[MT5TradeWithSetup]:
        """Holt alle offenen Trades."""
        trades, _ = await self.get_trades(status="open", limit=500, include_links=True)
        return trades

    def _parse_trade(self, data: dict) -> MT5Trade:
        """Parsed Trade-Daten."""
        return MT5Trade(
            trade_id=data["trade_id"],
            terminal_id=data["terminal_id"],
            ticket=data["ticket"],
            position_id=data.get("position_id"),
            symbol=data["symbol"],
            trade_type=MT5TradeType(data["trade_type"]),
            entry_time=data["entry_time"],
            entry_price=data["entry_price"],
            volume=data["volume"],
            exit_time=data.get("exit_time"),
            exit_price=data.get("exit_price"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            profit=data.get("profit"),
            profit_pips=data.get("profit_pips"),
            commission=data.get("commission"),
            swap=data.get("swap"),
            status=MT5TradeStatus(data.get("status", "open")),
            close_reason=data.get("close_reason"),
            magic_number=data.get("magic_number"),
            comment=data.get("comment"),
            timeframe=data.get("timeframe"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    # =========================================================================
    # Link-Operationen
    # =========================================================================

    async def get_trade_link(self, trade_id: str) -> Optional[MT5TradeSetupLink]:
        """Holt den Setup-Link für einen Trade."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/trades/{trade_id}/link")

            if response.status_code != 200:
                return None

            return self._parse_link(response.json())

        except Exception as e:
            logger.debug(f"No link found for trade {trade_id}: {e}")
            return None

    async def create_trade_link(self, link_data: dict) -> Optional[MT5TradeSetupLink]:
        """Erstellt einen Trade-Setup-Link."""
        try:
            client = await self._get_client()
            trade_id = link_data.get("trade_id")
            response = await client.post(
                f"{self._base_url}/trades/{trade_id}/link",
                json=link_data,
            )

            if response.status_code not in (200, 201):
                logger.warning(f"Failed to create link: {response.status_code}")
                return None

            return self._parse_link(response.json())

        except Exception as e:
            logger.error(f"Error creating link: {e}")
            return None

    async def delete_trade_link(self, trade_id: str) -> bool:
        """Löscht einen Trade-Setup-Link."""
        try:
            client = await self._get_client()
            response = await client.delete(f"{self._base_url}/trades/{trade_id}/link")
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Error deleting link: {e}")
            return False

    def _parse_link(self, data: dict) -> MT5TradeSetupLink:
        """Parsed Link-Daten."""
        from ..models.schemas import SignalDirection, ConfidenceLevel, MarketRegime, MT5LinkType, MT5OutcomeType

        return MT5TradeSetupLink(
            link_id=data["link_id"],
            trade_id=data["trade_id"],
            setup_symbol=data["setup_symbol"],
            setup_timeframe=data["setup_timeframe"],
            setup_timestamp=data["setup_timestamp"],
            setup_direction=SignalDirection(data["setup_direction"]),
            setup_score=data["setup_score"],
            setup_confidence=ConfidenceLevel(data["setup_confidence"]) if data.get("setup_confidence") else None,
            nhits_direction=SignalDirection(data["nhits_direction"]) if data.get("nhits_direction") else None,
            nhits_probability=data.get("nhits_probability"),
            hmm_regime=MarketRegime(data["hmm_regime"]) if data.get("hmm_regime") else None,
            hmm_score=data.get("hmm_score"),
            tcn_patterns=data.get("tcn_patterns"),
            tcn_confidence=data.get("tcn_confidence"),
            candlestick_patterns=data.get("candlestick_patterns"),
            candlestick_strength=data.get("candlestick_strength"),
            link_type=MT5LinkType(data.get("link_type", "auto")),
            link_confidence=data.get("link_confidence"),
            notes=data.get("notes"),
            followed_recommendation=data.get("followed_recommendation"),
            outcome_vs_prediction=MT5OutcomeType(data["outcome_vs_prediction"]) if data.get("outcome_vs_prediction") else None,
            created_at=data.get("created_at"),
        )

    # =========================================================================
    # Statistiken
    # =========================================================================

    async def get_trade_stats(
        self,
        terminal_id: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> dict:
        """Holt Trade-Statistiken vom Data Service."""
        try:
            client = await self._get_client()
            params = {}
            if terminal_id:
                params["terminal_id"] = terminal_id
            if symbol:
                params["symbol"] = symbol.upper()
            if since:
                params["since"] = since.isoformat()

            response = await client.get(
                f"{self._base_url}/trades/stats/summary",
                params=params,
            )

            if response.status_code != 200:
                logger.warning(f"Failed to fetch trade stats: {response.status_code}")
                return {}

            return response.json()

        except Exception as e:
            logger.error(f"Error fetching trade stats: {e}")
            return {}

    async def get_link_stats(self) -> dict:
        """Holt Link-Statistiken vom Data Service."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/links/stats")

            if response.status_code != 200:
                return {}

            return response.json()

        except Exception as e:
            logger.error(f"Error fetching link stats: {e}")
            return {}

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> dict:
        """Prüft die Verfügbarkeit des MT5 Endpoints im Data Service."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/health")

            if response.status_code == 200:
                return response.json()

            return {"status": "unavailable", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton-Instanz
mt5_trade_service = MT5TradeService()
