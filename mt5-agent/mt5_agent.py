"""
MT5 Trade Agent - Erfasst Trades aus MetaTrader 5 und sendet sie an den Data Service.

Dieses Script läuft auf der Windows-Maschine mit MT5 und überwacht alle
Trade-Aktivitäten. Neue und geschlossene Trades werden automatisch an den
Trading Workplace Service gemeldet.

Voraussetzungen:
    - MetaTrader 5 installiert und gestartet
    - Python 3.8+ mit MetaTrader5-Bibliothek
    - Netzwerkzugriff auf den Data Service (Port 3001)

Installation:
    pip install MetaTrader5 requests python-dotenv

Konfiguration:
    Erstellen Sie eine .env Datei oder setzen Sie Umgebungsvariablen:
    - DATA_SERVICE_URL: URL des Data Services (z.B. http://10.1.19.101:3001)
    - TERMINAL_API_KEY: API-Key vom registrierten Terminal
    - MT5_ACCOUNT: MT5 Kontonummer (optional, wird automatisch erkannt)
    - POLL_INTERVAL: Polling-Intervall in Sekunden (Standard: 5)

Verwendung:
    python mt5_agent.py

    Als Windows-Dienst:
    1. pip install pywin32
    2. python mt5_agent.py --install-service
    3. net start MT5TradeAgent
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 library not installed.")
    print("Install with: pip install MetaTrader5")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed.")
    print("Install with: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# =============================================================================
# Konfiguration
# =============================================================================

@dataclass
class AgentConfig:
    """Agent-Konfiguration."""
    data_service_url: str
    terminal_api_key: str
    terminal_id: str
    poll_interval: int = 5
    heartbeat_interval: int = 60
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Lädt Konfiguration aus Umgebungsvariablen."""
        return cls(
            data_service_url=os.getenv("DATA_SERVICE_URL", "http://localhost:3001"),
            terminal_api_key=os.getenv("TERMINAL_API_KEY", ""),
            terminal_id=os.getenv("TERMINAL_ID", ""),
            poll_interval=int(os.getenv("POLL_INTERVAL", "5")),
            heartbeat_interval=int(os.getenv("HEARTBEAT_INTERVAL", "60")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
        )


# =============================================================================
# Logging
# =============================================================================

def setup_logging(config: AgentConfig) -> logging.Logger:
    """Konfiguriert Logging."""
    logger = logging.getLogger("mt5_agent")
    logger.setLevel(getattr(logging, config.log_level))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional)
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# MT5 Agent
# =============================================================================

class MT5Agent:
    """
    MT5 Trade Agent.

    Überwacht MT5 Trades und sendet sie an den Data Service.
    """

    def __init__(self, config: AgentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._running = False
        self._known_tickets: set[int] = set()  # For tracking open positions
        self._processed_close_deals: set[int] = set()  # For tracking closed deals (separate!)
        self._known_positions: dict[int, dict] = {}
        self._last_heartbeat = 0

    def connect_mt5(self) -> bool:
        """Verbindet mit MT5."""
        if not mt5.initialize():
            self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return False

        self.logger.info(f"Connected to MT5: Account {account_info.login} @ {account_info.server}")
        self.logger.info(f"Balance: {account_info.balance} {account_info.currency}")
        return True

    def disconnect_mt5(self):
        """Trennt MT5-Verbindung."""
        mt5.shutdown()
        self.logger.info("MT5 disconnected")

    def start(self):
        """Startet den Agent."""
        self.logger.info("Starting MT5 Trade Agent...")

        if not self.connect_mt5():
            return False

        # Initiale Positionen laden
        self._load_initial_positions()

        # Bereits geschlossene Deals laden (damit sie nicht erneut gemeldet werden)
        self._load_existing_close_deals()

        self._running = True
        self.logger.info(f"Agent started. Polling interval: {self.config.poll_interval}s")

        try:
            while self._running:
                self._poll_cycle()
                time.sleep(self.config.poll_interval)
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested...")
        finally:
            self.stop()

        return True

    def stop(self):
        """Stoppt den Agent."""
        self._running = False
        self.disconnect_mt5()
        self.logger.info("Agent stopped")

    def _poll_cycle(self):
        """Ein Polling-Zyklus."""
        try:
            # Heartbeat senden
            now = time.time()
            if now - self._last_heartbeat > self.config.heartbeat_interval:
                self._send_heartbeat()
                self._last_heartbeat = now

            # Neue/geänderte Positionen prüfen
            self._check_positions()

            # Geschlossene Trades prüfen (aus History)
            self._check_history()

        except Exception as e:
            self.logger.error(f"Poll cycle error: {e}")

    def _load_initial_positions(self):
        """Lädt initiale offene Positionen."""
        positions = mt5.positions_get()
        if positions is None:
            return

        for pos in positions:
            self._known_positions[pos.ticket] = self._position_to_dict(pos)
            self._known_tickets.add(pos.ticket)

        self.logger.info(f"Loaded {len(positions)} open positions")

    def _load_existing_close_deals(self):
        """Synchronisiert geschlossene Trades beim Start und markiert sie als verarbeitet."""
        from datetime import timedelta

        # Letzte 7 Tage (nicht nur 24 Stunden)
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=7)

        self.logger.info(f"Fetching deal history from {from_date} to {to_date}")

        # MT5 erwartet datetime-Objekte
        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            self.logger.info(f"No history deals found (error: {mt5.last_error()})")
            return

        self.logger.info(f"Total deals in history: {len(deals)}")

        # Debug: Zeige alle Deals mit INFO-Level
        for d in deals:
            entry_type = "IN" if d.entry == mt5.DEAL_ENTRY_IN else ("OUT" if d.entry == mt5.DEAL_ENTRY_OUT else f"OTHER({d.entry})")
            self.logger.info(f"  Deal: ticket={d.ticket} position={d.position_id} entry={entry_type} symbol={d.symbol} time={datetime.fromtimestamp(d.time)}")

        close_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        self.logger.info(f"Found {len(close_deals)} close deals (DEAL_ENTRY_OUT)")

        # Zeige Close-Deals Details
        for d in close_deals:
            self.logger.info(f"  Close deal: deal_ticket={d.ticket} position_id={d.position_id} symbol={d.symbol}")

        # Hole Trades aus der DB, um zu prüfen welche noch nicht synchronisiert sind
        synced_count = 0
        for deal in close_deals:
            # Immer als verarbeitet markieren
            self._processed_close_deals.add(deal.ticket)

            # Prüfe ob Trade in DB noch "open" ist und synchronisiere
            trade_id = self._get_trade_id(deal.position_id)
            if trade_id:
                # Trade existiert - sende Close-Update
                self.logger.info(f"Syncing closed trade: position={deal.position_id}")
                self._report_closed_trade(deal)
                synced_count += 1

        self.logger.info(f"Startup sync complete: {synced_count} trades synchronized, {len(close_deals)} close deals tracked")

    def _check_positions(self):
        """Prüft auf neue oder geänderte Positionen."""
        positions = mt5.positions_get()
        if positions is None:
            return

        current_tickets = set()

        for pos in positions:
            current_tickets.add(pos.ticket)

            # Neue Position?
            if pos.ticket not in self._known_tickets:
                self.logger.info(f"New position detected: {pos.ticket} {pos.symbol}")
                self._report_new_trade(pos)
                self._known_tickets.add(pos.ticket)
                self._known_positions[pos.ticket] = self._position_to_dict(pos)

            # SL/TP geändert?
            elif pos.ticket in self._known_positions:
                old = self._known_positions[pos.ticket]
                if pos.sl != old.get("sl") or pos.tp != old.get("tp"):
                    self.logger.info(f"Position modified: {pos.ticket} SL/TP changed")
                    self._report_trade_update(pos)
                    self._known_positions[pos.ticket] = self._position_to_dict(pos)

        # Geschlossene Positionen erkennen
        closed_tickets = self._known_tickets - current_tickets
        for ticket in closed_tickets:
            self._known_tickets.discard(ticket)
            if ticket in self._known_positions:
                del self._known_positions[ticket]
            # Trade wird über History erkannt und gemeldet

    def _check_history(self):
        """Prüft Trade-History auf geschlossene Trades."""
        from datetime import timedelta

        # Letzte 24 Stunden
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(hours=24)

        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            return

        for deal in deals:
            # Nur OUT-Deals (Trade geschlossen)
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue

            # Verwende deal.ticket für Tracking (eindeutig pro Close-Deal)
            deal_ticket = deal.ticket

            # Bereits verarbeitet? Nur prüfen im separaten Close-Deals Set
            if deal_ticket in self._processed_close_deals:
                continue

            # Als verarbeitet markieren
            self._processed_close_deals.add(deal_ticket)
            self.logger.info(f"Closed trade detected: position={deal.position_id} deal={deal_ticket} {deal.symbol}")
            self._report_closed_trade(deal)

    def _report_new_trade(self, position):
        """Meldet einen neuen Trade an den Data Service."""
        trade_data = {
            "terminal_id": self.config.terminal_id,
            "ticket": position.ticket,
            "position_id": position.ticket,  # In MT5 gleich dem Ticket
            "symbol": position.symbol,
            "trade_type": "buy" if position.type == mt5.POSITION_TYPE_BUY else "sell",
            "entry_time": datetime.fromtimestamp(position.time, tz=timezone.utc).isoformat(),
            "entry_price": position.price_open,
            "volume": position.volume,
            "stop_loss": position.sl if position.sl > 0 else None,
            "take_profit": position.tp if position.tp > 0 else None,
            "magic_number": position.magic if position.magic > 0 else None,
            "comment": position.comment if position.comment else None,
        }

        self._send_trade(trade_data)

    def _report_trade_update(self, position):
        """Meldet eine Trade-Änderung (SL/TP)."""
        # Finde trade_id für dieses Ticket
        trade_id = self._get_trade_id(position.ticket)
        if not trade_id:
            self.logger.warning(f"Trade ID not found for ticket {position.ticket}")
            return

        update_data = {
            "stop_loss": position.sl if position.sl > 0 else None,
            "take_profit": position.tp if position.tp > 0 else None,
        }

        self._send_trade_update(trade_id, update_data)

    def _report_closed_trade(self, deal):
        """Meldet einen geschlossenen Trade."""
        # Position-Ticket finden
        position_ticket = deal.position_id

        # Trade-ID ermitteln
        self.logger.info(f"Looking up trade_id for position {position_ticket}...")
        trade_id = self._get_trade_id(position_ticket)
        if not trade_id:
            # Neuen Trade + Close melden
            self.logger.warning(f"Trade not found in DB for position {position_ticket} - creating new record")
            trade_data = {
                "terminal_id": self.config.terminal_id,
                "ticket": position_ticket,
                "position_id": position_ticket,
                "symbol": deal.symbol,
                "trade_type": "buy" if deal.type == mt5.DEAL_TYPE_BUY else "sell",
                "entry_time": datetime.fromtimestamp(deal.time, tz=timezone.utc).isoformat(),
                "entry_price": deal.price,
                "volume": deal.volume,
            }
            response = self._send_trade(trade_data)
            if response:
                trade_id = response.get("trade_id")

        if trade_id:
            # Bestimme Close-Reason
            close_reason = "manual"
            if deal.reason == mt5.DEAL_REASON_SL:
                close_reason = "sl"
            elif deal.reason == mt5.DEAL_REASON_TP:
                close_reason = "tp"
            elif deal.reason == mt5.DEAL_REASON_SO:
                close_reason = "margin"

            update_data = {
                "exit_time": datetime.fromtimestamp(deal.time, tz=timezone.utc).isoformat(),
                "exit_price": deal.price,
                "profit": deal.profit,
                "commission": deal.commission,
                "swap": deal.swap,
                "status": "closed",
                "close_reason": close_reason,
            }

            self._send_trade_update(trade_id, update_data)
        else:
            self.logger.error(f"Could not close trade - no trade_id found for position {deal.position_id}")

    def _send_trade(self, trade_data: dict) -> Optional[dict]:
        """Sendet Trade-Daten an den Data Service."""
        url = f"{self.config.data_service_url}/api/v1/mt5/trades"
        headers = {"X-API-Key": self.config.terminal_api_key}

        try:
            response = requests.post(url, json=trade_data, headers=headers, timeout=10)
            if response.status_code in (200, 201):
                self.logger.debug(f"Trade reported: {trade_data.get('ticket')}")
                return response.json()
            else:
                self.logger.error(f"Failed to report trade: {response.status_code} {response.text}")
                return None
        except requests.RequestException as e:
            self.logger.error(f"HTTP error reporting trade: {e}")
            return None

    def _send_trade_update(self, trade_id: str, update_data: dict):
        """Sendet Trade-Update an den Data Service."""
        url = f"{self.config.data_service_url}/api/v1/mt5/trades/{trade_id}"
        headers = {"X-API-Key": self.config.terminal_api_key}

        try:
            response = requests.put(url, json=update_data, headers=headers, timeout=10)
            if response.status_code == 200:
                self.logger.info(f"Trade updated successfully: {trade_id} -> {update_data.get('status', 'modified')}")
            else:
                self.logger.error(f"Failed to update trade: {response.status_code} {response.text}")
        except requests.RequestException as e:
            self.logger.error(f"HTTP error updating trade: {e}")

    def _send_heartbeat(self):
        """Sendet Heartbeat an den Data Service."""
        url = f"{self.config.data_service_url}/api/v1/mt5/terminals/{self.config.terminal_id}/heartbeat"
        headers = {"X-API-Key": self.config.terminal_api_key}

        try:
            response = requests.post(url, headers=headers, timeout=5)
            if response.status_code == 200:
                self.logger.debug("Heartbeat sent")
            else:
                self.logger.warning(f"Heartbeat failed: {response.status_code}")
        except requests.RequestException as e:
            self.logger.warning(f"Heartbeat error: {e}")

    def _get_trade_id(self, ticket: int) -> Optional[str]:
        """Ermittelt die Trade-ID für ein Ticket vom Data Service."""
        url = f"{self.config.data_service_url}/api/v1/mt5/trades"
        params = {
            "terminal_id": self.config.terminal_id,
            "limit": 1000,
        }
        headers = {"X-API-Key": self.config.terminal_api_key}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for trade in data.get("trades", []):
                    if trade.get("ticket") == ticket:
                        return trade.get("trade_id")
        except requests.RequestException:
            pass

        return None

    def _position_to_dict(self, position) -> dict:
        """Konvertiert eine MT5-Position zu einem Dictionary."""
        return {
            "ticket": position.ticket,
            "symbol": position.symbol,
            "type": position.type,
            "volume": position.volume,
            "price_open": position.price_open,
            "sl": position.sl,
            "tp": position.tp,
            "profit": position.profit,
            "time": position.time,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(description="MT5 Trade Agent")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Pfad zur Konfigurationsdatei (.env)"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Neues Terminal registrieren"
    )

    args = parser.parse_args()

    # Konfiguration laden
    if args.config:
        load_dotenv(args.config)

    config = AgentConfig.from_env()
    logger = setup_logging(config)

    # Terminal registrieren?
    if args.register:
        register_terminal(config, logger)
        return

    # Konfiguration prüfen
    if not config.terminal_api_key or not config.terminal_id:
        logger.error("TERMINAL_API_KEY and TERMINAL_ID must be set.")
        logger.info("Run with --register to register a new terminal first.")
        sys.exit(1)

    # Agent starten
    agent = MT5Agent(config, logger)
    agent.start()


def register_terminal(config: AgentConfig, logger: logging.Logger):
    """Registriert ein neues Terminal beim Data Service."""
    if not mt5.initialize():
        logger.error(f"MT5 initialize failed: {mt5.last_error()}")
        return

    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        mt5.shutdown()
        return

    # Terminal-Daten
    terminal_data = {
        "name": f"MT5-{account_info.login}",
        "account_number": account_info.login,
        "broker_name": account_info.company,
        "server": account_info.server,
        "account_type": "demo" if account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "real",
        "currency": account_info.currency,
        "leverage": account_info.leverage,
    }

    mt5.shutdown()

    # An Data Service senden
    url = f"{config.data_service_url}/api/v1/mt5/terminals"
    try:
        response = requests.post(url, json=terminal_data, timeout=10)
        if response.status_code in (200, 201):
            data = response.json()
            logger.info("=" * 60)
            logger.info("TERMINAL ERFOLGREICH REGISTRIERT!")
            logger.info("=" * 60)
            logger.info(f"Terminal ID: {data.get('terminal_id')}")
            logger.info(f"API Key:     {data.get('api_key')}")
            logger.info("=" * 60)
            logger.info("Fügen Sie diese Werte zu Ihrer .env Datei hinzu:")
            logger.info(f"TERMINAL_ID={data.get('terminal_id')}")
            logger.info(f"TERMINAL_API_KEY={data.get('api_key')}")
            logger.info("=" * 60)
        else:
            logger.error(f"Registration failed: {response.status_code} {response.text}")
    except requests.RequestException as e:
        logger.error(f"HTTP error during registration: {e}")


if __name__ == "__main__":
    main()
