"""MT5 Performance Service.

Berechnet Performance-Metriken für MT5 Trades inkl. Setup-Analyse.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict

from loguru import logger

from ..models.schemas import (
    MT5PerformanceMetrics,
    MT5TradeWithSetup,
    MT5TradeStatus,
    MT5TradeType,
    MT5OutcomeType,
)
from .mt5_trade_service import mt5_trade_service


class MT5PerformanceService:
    """Service für Performance-Analyse von MT5 Trades."""

    async def calculate_metrics(
        self,
        terminal_id: Optional[str] = None,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> MT5PerformanceMetrics:
        """
        Berechnet umfassende Performance-Metriken.

        Args:
            terminal_id: Optional - Filter nach Terminal
            symbol: Optional - Filter nach Symbol
            since: Optional - Startdatum
            until: Optional - Enddatum

        Returns:
            MT5PerformanceMetrics mit allen berechneten Werten
        """
        # Basis-Statistiken vom Data Service
        base_stats = await mt5_trade_service.get_trade_stats(
            terminal_id=terminal_id,
            symbol=symbol,
            since=since,
        )

        # Link-Statistiken
        link_stats = await mt5_trade_service.get_link_stats()

        # Trades für Detail-Berechnungen holen
        trades, total = await mt5_trade_service.get_trades(
            terminal_id=terminal_id,
            symbol=symbol,
            since=since,
            until=until,
            limit=1000,
            include_links=True,
        )

        # Aufschlüsselungen berechnen
        trades_by_symbol = defaultdict(int)
        trades_by_direction = defaultdict(int)
        profit_by_symbol = defaultdict(float)

        trades_with_setup = 0
        trades_following_setup = 0
        profit_following_setup = 0.0
        profit_against_setup = 0.0

        winning_profits = []
        losing_profits = []

        for trade in trades:
            # Symbol-Aufschlüsselung
            trades_by_symbol[trade.symbol] += 1

            # Richtungs-Aufschlüsselung
            direction = "buy" if trade.trade_type == MT5TradeType.BUY else "sell"
            trades_by_direction[direction] += 1

            # Profit pro Symbol
            if trade.profit is not None:
                profit_by_symbol[trade.symbol] += trade.profit

                # Winning/Losing Profits für Durchschnitte
                if trade.status == MT5TradeStatus.CLOSED:
                    if trade.profit > 0:
                        winning_profits.append(trade.profit)
                    elif trade.profit < 0:
                        losing_profits.append(trade.profit)

            # Setup-Analyse
            if trade.setup:
                trades_with_setup += 1

                # Prüfen ob Trade der Empfehlung folgte
                if trade.setup.followed_recommendation is True:
                    trades_following_setup += 1
                    if trade.profit is not None:
                        profit_following_setup += trade.profit
                elif trade.setup.followed_recommendation is False:
                    if trade.profit is not None:
                        profit_against_setup += trade.profit

        # Durchschnittswerte berechnen
        average_win = sum(winning_profits) / len(winning_profits) if winning_profits else None
        average_loss = sum(losing_profits) / len(losing_profits) if losing_profits else None

        # Setup-Folgerate
        setup_follow_rate = 0.0
        if trades_with_setup > 0:
            setup_follow_rate = (trades_following_setup / trades_with_setup) * 100

        # Vorhersage-Genauigkeit aus Link-Stats
        setup_prediction_accuracy = link_stats.get("prediction_accuracy", 0.0)

        return MT5PerformanceMetrics(
            # Basis-Statistiken
            total_trades=base_stats.get("total_trades", 0),
            open_trades=base_stats.get("open_trades", 0),
            closed_trades=base_stats.get("closed_trades", 0),
            winning_trades=base_stats.get("winning_trades", 0),
            losing_trades=base_stats.get("losing_trades", 0),
            win_rate=base_stats.get("win_rate", 0.0),

            # Profit-Metriken
            total_profit=base_stats.get("total_profit", 0.0),
            total_loss=base_stats.get("total_loss", 0.0),
            net_profit=base_stats.get("net_profit", 0.0),
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=base_stats.get("profit_factor", 0.0),
            max_drawdown=None,  # TODO: Drawdown-Berechnung

            # Kosten
            total_commission=base_stats.get("total_commission", 0.0),
            total_swap=base_stats.get("total_swap", 0.0),

            # Setup-verknüpfte Metriken
            trades_with_setup=trades_with_setup,
            trades_following_setup=trades_following_setup,
            setup_follow_rate=setup_follow_rate,
            profit_following_setup=profit_following_setup,
            profit_against_setup=profit_against_setup,
            setup_prediction_accuracy=setup_prediction_accuracy,

            # Aufschlüsselungen
            trades_by_symbol=dict(trades_by_symbol),
            trades_by_direction=dict(trades_by_direction),
            profit_by_symbol=dict(profit_by_symbol),
        )

    async def calculate_setup_performance(
        self,
        since: Optional[datetime] = None,
    ) -> dict:
        """
        Analysiert die Performance der Trading-Setups.

        Zeigt wie profitabel Trades sind die Setups folgen vs. nicht folgen.
        """
        link_stats = await mt5_trade_service.get_link_stats()

        # Trades mit Links holen
        trades, _ = await mt5_trade_service.get_trades(
            since=since,
            limit=1000,
            include_links=True,
        )

        # Kategorisieren
        followed_trades = []
        against_trades = []
        no_link_trades = []

        for trade in trades:
            if not trade.setup:
                no_link_trades.append(trade)
            elif trade.setup.followed_recommendation is True:
                followed_trades.append(trade)
            elif trade.setup.followed_recommendation is False:
                against_trades.append(trade)

        def calc_stats(trade_list: list[MT5TradeWithSetup]) -> dict:
            """Berechnet Statistiken für eine Trade-Liste."""
            if not trade_list:
                return {
                    "count": 0,
                    "win_rate": 0.0,
                    "total_profit": 0.0,
                    "avg_profit": 0.0,
                }

            closed = [t for t in trade_list if t.status == MT5TradeStatus.CLOSED]
            if not closed:
                return {
                    "count": len(trade_list),
                    "win_rate": 0.0,
                    "total_profit": 0.0,
                    "avg_profit": 0.0,
                }

            wins = [t for t in closed if t.profit and t.profit > 0]
            total_profit = sum(t.profit or 0 for t in closed)

            return {
                "count": len(closed),
                "win_rate": (len(wins) / len(closed)) * 100 if closed else 0.0,
                "total_profit": total_profit,
                "avg_profit": total_profit / len(closed) if closed else 0.0,
            }

        return {
            "followed_setup": calc_stats(followed_trades),
            "against_setup": calc_stats(against_trades),
            "no_setup_link": calc_stats(no_link_trades),
            "link_stats": link_stats,
            "recommendation": self._generate_recommendation(
                followed=calc_stats(followed_trades),
                against=calc_stats(against_trades),
            ),
        }

    async def get_symbol_breakdown(
        self,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Aufschlüsselung der Performance nach Symbol.

        Returns:
            Liste mit Performance-Daten pro Symbol
        """
        trades, _ = await mt5_trade_service.get_trades(
            since=since,
            limit=1000,
            include_links=False,
        )

        # Nach Symbol gruppieren
        by_symbol = defaultdict(list)
        for trade in trades:
            by_symbol[trade.symbol].append(trade)

        result = []
        for symbol, symbol_trades in by_symbol.items():
            closed = [t for t in symbol_trades if t.status == MT5TradeStatus.CLOSED]
            if not closed:
                continue

            wins = [t for t in closed if t.profit and t.profit > 0]
            total_profit = sum(t.profit or 0 for t in closed)

            result.append({
                "symbol": symbol,
                "total_trades": len(symbol_trades),
                "closed_trades": len(closed),
                "winning_trades": len(wins),
                "win_rate": (len(wins) / len(closed)) * 100 if closed else 0.0,
                "total_profit": total_profit,
                "avg_profit": total_profit / len(closed) if closed else 0.0,
            })

        # Sortieren nach Profit
        result.sort(key=lambda x: x["total_profit"], reverse=True)
        return result

    async def get_time_analysis(
        self,
        since: Optional[datetime] = None,
        group_by: str = "day",  # day, week, month
    ) -> list[dict]:
        """
        Zeitliche Analyse der Trading-Performance.

        Args:
            since: Startdatum
            group_by: Gruppierung (day, week, month)

        Returns:
            Liste mit Performance-Daten pro Zeitperiode
        """
        trades, _ = await mt5_trade_service.get_trades(
            since=since,
            limit=1000,
            include_links=False,
        )

        # Nach Zeit gruppieren
        by_period = defaultdict(list)

        for trade in trades:
            if trade.status != MT5TradeStatus.CLOSED:
                continue

            entry = trade.entry_time
            if isinstance(entry, str):
                entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))

            if group_by == "day":
                key = entry.strftime("%Y-%m-%d")
            elif group_by == "week":
                key = entry.strftime("%Y-W%W")
            else:  # month
                key = entry.strftime("%Y-%m")

            by_period[key].append(trade)

        result = []
        for period, period_trades in sorted(by_period.items()):
            wins = [t for t in period_trades if t.profit and t.profit > 0]
            total_profit = sum(t.profit or 0 for t in period_trades)

            result.append({
                "period": period,
                "trades": len(period_trades),
                "wins": len(wins),
                "losses": len(period_trades) - len(wins),
                "win_rate": (len(wins) / len(period_trades)) * 100 if period_trades else 0.0,
                "profit": total_profit,
            })

        return result

    def _generate_recommendation(self, followed: dict, against: dict) -> str:
        """Generiert eine Empfehlung basierend auf Setup-Performance."""
        if followed["count"] < 10 or against["count"] < 5:
            return "Nicht genügend Daten für eine Empfehlung. Mindestens 10 Trades mit Setup-Befolgung und 5 dagegen benötigt."

        # Vergleich
        follow_better = followed["win_rate"] > against["win_rate"]
        profit_better = followed["total_profit"] > against["total_profit"]

        if follow_better and profit_better:
            diff_winrate = followed["win_rate"] - against["win_rate"]
            return (
                f"Setup-Befolgung empfohlen. "
                f"Win-Rate ist {diff_winrate:.1f}% höher und Profit ist besser "
                f"wenn den Setup-Empfehlungen gefolgt wird."
            )
        elif follow_better:
            return (
                f"Gemischte Ergebnisse. "
                f"Win-Rate ist bei Setup-Befolgung höher, aber Gesamtprofit niedriger. "
                f"Prüfen Sie die Positionsgrössen."
            )
        elif profit_better:
            return (
                f"Gemischte Ergebnisse. "
                f"Gesamtprofit ist bei Setup-Befolgung höher, aber Win-Rate niedriger. "
                f"Gutes Risikomanagement bei Setup-Trades."
            )
        else:
            return (
                f"Setup-Empfehlungen zeigen aktuell keine Verbesserung. "
                f"Prüfen Sie die Setup-Parameter oder Ihren Trading-Stil."
            )


# Singleton-Instanz
mt5_performance_service = MT5PerformanceService()
