"""
Level Calculator Service.

Berechnet Entry/Exit-Levels (Entry, SL, TP1, TP2, TP3) basierend auf ATR.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from ..config import settings
from ..models.schemas import EntryExitLevels, SignalDirection


@dataclass
class LevelConfig:
    """Konfiguration für ATR-Multiplikatoren."""
    sl_multiplier: float
    tp1_multiplier: float
    tp2_multiplier: float
    tp3_multiplier: float


class LevelCalculatorService:
    """Berechnet Entry/Exit-Levels basierend auf ATR und Richtung."""

    # Asset-Klassen-spezifische Multiplikatoren
    ASSET_CONFIGS = {
        "crypto": LevelConfig(
            sl_multiplier=2.0,
            tp1_multiplier=2.0,
            tp2_multiplier=3.5,
            tp3_multiplier=5.0,
        ),
        "forex": LevelConfig(
            sl_multiplier=1.5,
            tp1_multiplier=1.5,
            tp2_multiplier=2.5,
            tp3_multiplier=4.0,
        ),
        "index": LevelConfig(
            sl_multiplier=1.5,
            tp1_multiplier=1.5,
            tp2_multiplier=2.5,
            tp3_multiplier=4.0,
        ),
        "commodity": LevelConfig(
            sl_multiplier=2.0,
            tp1_multiplier=2.0,
            tp2_multiplier=3.0,
            tp3_multiplier=4.5,
        ),
        "default": LevelConfig(
            sl_multiplier=1.5,
            tp1_multiplier=1.5,
            tp2_multiplier=2.5,
            tp3_multiplier=4.0,
        ),
    }

    # Crypto-Symbole erkennen
    CRYPTO_BASES = {
        "BTC", "ETH", "XRP", "LTC", "ADA", "DOT", "SOL", "DOGE", "AVAX",
        "MATIC", "LINK", "UNI", "ATOM", "XLM", "ALGO", "FIL", "VET", "AAVE",
        "EOS", "XTZ", "THETA", "NEO", "KLAY", "FLOW", "XMR", "SAND", "MANA",
        "AXS", "CHZ", "ENJ", "SUSHI", "COMP", "SNX", "CRV", "YFI", "MKR",
        "SHIB", "APE", "GMT", "NEAR", "FTM", "ONE", "HBAR", "EGLD", "ICP",
    }

    # Forex-Währungen
    FOREX_CURRENCIES = {
        "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK",
        "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY", "PLN", "CZK", "HUF", "RON",
    }

    # Indizes
    INDICES = {
        "US30", "US500", "US100", "NAS100", "GER40", "UK100", "FRA40", "ESP35",
        "JP225", "AUS200", "HK50", "EURO50", "DAX", "FTSE", "CAC", "IBEX",
        "NIKKEI", "HANGSENG", "SPX", "NDX", "DJI", "VIX", "DE40", "DE30",
    }

    # Rohstoffe
    COMMODITIES = {
        "XAUUSD", "XAGUSD", "GOLD", "SILVER", "OIL", "WTICOUSD", "BCOUSD",
        "WTI", "BRENT", "NATGAS", "NGAS", "COPPER", "PLATINUM", "PALLADIUM",
        "XPTUSD", "XPDUSD", "WHEAT", "CORN", "SOYBEAN", "COFFEE", "SUGAR",
    }

    def calculate_levels(
        self,
        current_price: float,
        direction: SignalDirection,
        atr: float,
        symbol: Optional[str] = None,
    ) -> Optional[EntryExitLevels]:
        """
        Berechnet Entry/Exit-Levels basierend auf ATR.

        Args:
            current_price: Aktueller Marktpreis
            direction: LONG oder SHORT
            atr: Average True Range (14)
            symbol: Symbol für Asset-Klassen-Erkennung

        Returns:
            EntryExitLevels oder None bei NEUTRAL oder ungültigen Daten
        """
        # Keine Levels für neutrale Signale
        if direction == SignalDirection.NEUTRAL:
            return None

        # Validierung
        if current_price <= 0 or atr <= 0:
            logger.warning(f"Invalid price ({current_price}) or ATR ({atr})")
            return None

        # Config für Asset-Klasse holen
        config = self._get_config_for_symbol(symbol)

        # Entry = aktueller Preis
        entry_price = current_price

        # Levels berechnen basierend auf Richtung
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - (atr * config.sl_multiplier)
            take_profit_1 = entry_price + (atr * config.tp1_multiplier)
            take_profit_2 = entry_price + (atr * config.tp2_multiplier)
            take_profit_3 = entry_price + (atr * config.tp3_multiplier)
        else:  # SHORT
            stop_loss = entry_price + (atr * config.sl_multiplier)
            take_profit_1 = entry_price - (atr * config.tp1_multiplier)
            take_profit_2 = entry_price - (atr * config.tp2_multiplier)
            take_profit_3 = entry_price - (atr * config.tp3_multiplier)

        # Risk/Reward Ratio berechnen
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit_1 - entry_price)
        risk_reward_ratio = round(reward / risk, 2) if risk > 0 else None

        # Preise runden basierend auf Symbol
        decimals = self._get_price_decimals(symbol, current_price)
        entry_price = round(entry_price, decimals)
        stop_loss = round(stop_loss, decimals)
        take_profit_1 = round(take_profit_1, decimals)
        take_profit_2 = round(take_profit_2, decimals)
        take_profit_3 = round(take_profit_3, decimals)

        return EntryExitLevels(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            risk_reward_ratio=risk_reward_ratio,
        )

    def calculate_levels_percentage_fallback(
        self,
        current_price: float,
        direction: SignalDirection,
        symbol: Optional[str] = None,
    ) -> Optional[EntryExitLevels]:
        """
        Fallback-Berechnung mit Prozent-Werten wenn ATR nicht verfügbar.

        Verwendet feste Prozentsätze basierend auf Asset-Klasse.
        """
        if direction == SignalDirection.NEUTRAL:
            return None

        if current_price <= 0:
            return None

        # Asset-Klasse bestimmen
        asset_class = self._detect_asset_class(symbol)

        # Prozent-Multiplikatoren pro Asset-Klasse
        pct_configs = {
            "crypto": (0.02, 0.02, 0.035, 0.05),      # 2%, 3.5%, 5%
            "forex": (0.005, 0.005, 0.01, 0.015),     # 0.5%, 1%, 1.5%
            "index": (0.01, 0.01, 0.02, 0.03),        # 1%, 2%, 3%
            "commodity": (0.015, 0.015, 0.025, 0.04), # 1.5%, 2.5%, 4%
            "default": (0.01, 0.01, 0.02, 0.03),      # 1%, 2%, 3%
        }

        sl_pct, tp1_pct, tp2_pct, tp3_pct = pct_configs.get(
            asset_class, pct_configs["default"]
        )

        entry_price = current_price

        if direction == SignalDirection.LONG:
            stop_loss = entry_price * (1 - sl_pct)
            take_profit_1 = entry_price * (1 + tp1_pct)
            take_profit_2 = entry_price * (1 + tp2_pct)
            take_profit_3 = entry_price * (1 + tp3_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + sl_pct)
            take_profit_1 = entry_price * (1 - tp1_pct)
            take_profit_2 = entry_price * (1 - tp2_pct)
            take_profit_3 = entry_price * (1 - tp3_pct)

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit_1 - entry_price)
        risk_reward_ratio = round(reward / risk, 2) if risk > 0 else None

        decimals = self._get_price_decimals(symbol, current_price)
        entry_price = round(entry_price, decimals)
        stop_loss = round(stop_loss, decimals)
        take_profit_1 = round(take_profit_1, decimals)
        take_profit_2 = round(take_profit_2, decimals)
        take_profit_3 = round(take_profit_3, decimals)

        return EntryExitLevels(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            risk_reward_ratio=risk_reward_ratio,
        )

    def _detect_asset_class(self, symbol: Optional[str]) -> str:
        """Erkennt die Asset-Klasse eines Symbols."""
        if not symbol:
            return "default"

        symbol_upper = symbol.upper()

        # Crypto erkennen
        for base in self.CRYPTO_BASES:
            if symbol_upper.startswith(base):
                return "crypto"

        # Commodities erkennen
        if symbol_upper in self.COMMODITIES:
            return "commodity"
        if symbol_upper.startswith("XAU") or symbol_upper.startswith("XAG"):
            return "commodity"

        # Indizes erkennen
        if symbol_upper in self.INDICES:
            return "index"
        for idx in self.INDICES:
            if symbol_upper.startswith(idx):
                return "index"

        # Forex erkennen (6 Zeichen, beide Teile sind Währungen)
        if len(symbol_upper) == 6:
            base = symbol_upper[:3]
            quote = symbol_upper[3:]
            if base in self.FOREX_CURRENCIES and quote in self.FOREX_CURRENCIES:
                return "forex"

        return "default"

    def _get_config_for_symbol(self, symbol: Optional[str]) -> LevelConfig:
        """Gibt die Level-Konfiguration für ein Symbol zurück."""
        asset_class = self._detect_asset_class(symbol)
        return self.ASSET_CONFIGS.get(asset_class, self.ASSET_CONFIGS["default"])

    def _get_price_decimals(
        self, symbol: Optional[str], price: float
    ) -> int:
        """Bestimmt die Anzahl Dezimalstellen für ein Symbol."""
        if not symbol:
            # Basierend auf Preis schätzen
            if price > 10000:
                return 2  # BTC, Indizes
            elif price > 100:
                return 2
            elif price > 1:
                return 4
            else:
                return 6

        symbol_upper = symbol.upper()

        # JPY-Paare: 3 Dezimalen
        if symbol_upper.endswith("JPY"):
            return 3

        # Forex: 5 Dezimalen
        if self._detect_asset_class(symbol) == "forex":
            return 5

        # Crypto: abhängig vom Preis
        if self._detect_asset_class(symbol) == "crypto":
            if price > 1000:
                return 2
            elif price > 10:
                return 4
            elif price > 1:
                return 5
            else:
                return 8

        # Indizes & Commodities: 2 Dezimalen
        if self._detect_asset_class(symbol) in ["index", "commodity"]:
            return 2

        # Default
        return 2


# Singleton-Instanz
level_calculator = LevelCalculatorService()
