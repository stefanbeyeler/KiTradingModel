"""Telegram Notifier für Watchdog Alerts."""

from datetime import datetime, timezone
from typing import List, Optional

import httpx
from loguru import logger


class TelegramNotifier:
    """Sendet Telegram-Nachrichten über die Bot API."""

    def __init__(self, bot_token: str, chat_ids: str):
        """
        Initialisiert den Telegram Notifier.

        Args:
            bot_token: Bot-Token von @BotFather
            chat_ids: Kommagetrennte Chat-IDs
        """
        self.bot_token = bot_token
        self.chat_ids = self._parse_chat_ids(chat_ids)
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.enabled = bool(bot_token) and bool(self.chat_ids)

        if self.enabled:
            logger.info(f"Telegram notifier initialized with {len(self.chat_ids)} recipients")
        else:
            logger.warning("Telegram notifier disabled - missing token or chat IDs")

    def _parse_chat_ids(self, chat_ids_str: str) -> List[str]:
        """Parst kommagetrennte Chat-IDs."""
        if not chat_ids_str:
            return []
        return [cid.strip() for cid in chat_ids_str.split(",") if cid.strip()]

    def _escape_markdown(self, text: str) -> str:
        """Escaped Sonderzeichen für Telegram MarkdownV2."""
        special_chars = [
            "_", "*", "[", "]", "(", ")", "~", "`", ">",
            "#", "+", "-", "=", "|", "{", "}", ".", "!"
        ]
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    async def send_alert(
        self,
        service_name: str,
        state: str,
        error: Optional[str] = None,
        recovery: bool = False
    ) -> bool:
        """
        Sendet einen Alert an alle konfigurierten Chats.

        Args:
            service_name: Name des betroffenen Services
            state: Aktueller Status (UNHEALTHY, DEGRADED, etc.)
            error: Optionale Fehlermeldung
            recovery: True wenn es eine Wiederherstellungsmeldung ist

        Returns:
            True wenn mindestens eine Nachricht erfolgreich gesendet wurde
        """
        if not self.enabled:
            logger.debug("Telegram notifications disabled")
            return False

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if recovery:
            message = (
                f"✅ *SERVICE RECOVERED*\n\n"
                f"📦 Service: `{service_name}`\n"
                f"🕐 Zeit: {self._escape_markdown(timestamp)}\n"
                f"📊 Status: {state}\n\n"
                f"Der Service ist wieder verfügbar\\."
            )
        else:
            emoji = "🚨" if state == "UNHEALTHY" else "⚠️"
            error_escaped = self._escape_markdown(error or "Keine Details")
            message = (
                f"{emoji} *SERVICE ALERT*\n\n"
                f"📦 Service: `{service_name}`\n"
                f"🕐 Zeit: {self._escape_markdown(timestamp)}\n"
                f"📊 Status: *{state}*\n"
                f"❌ Fehler: {error_escaped}\n\n"
                f"Bitte umgehend prüfen\\!"
            )

        return await self._send_message(message, parse_mode="MarkdownV2")

    async def send_daily_summary(self, summary: dict) -> bool:
        """Sendet eine tägliche Zusammenfassung."""
        if not self.enabled:
            return False

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        healthy = summary.get("healthy", 0)
        degraded = summary.get("degraded", 0)
        unhealthy = summary.get("unhealthy", 0)
        total = healthy + degraded + unhealthy

        status_emoji = "✅" if unhealthy == 0 else "⚠️"

        message = (
            f"📊 *DAILY SYSTEM REPORT*\n\n"
            f"🕐 Zeit: {timestamp}\n\n"
            f"{status_emoji} Service Status:\n"
            f"• Healthy: {healthy}/{total}\n"
            f"• Degraded: {degraded}/{total}\n"
            f"• Unhealthy: {unhealthy}/{total}\n\n"
            f"📈 24h Statistiken:\n"
            f"• Alerts: {summary.get('alerts_24h', 0)}\n"
            f"• Avg Response: {summary.get('avg_response_ms', 0):.0f}ms\n"
            f"• Uptime: {summary.get('uptime_percent', 0):.1f}%"
        )

        return await self._send_message(message, parse_mode="Markdown")

    async def send_test_message(self) -> dict:
        """Sendet eine Test-Nachricht und gibt Ergebnis zurück."""
        results = {"success": [], "failed": [], "enabled": self.enabled}

        if not self.enabled:
            results["error"] = "Telegram not configured"
            return results

        message = (
            "🧪 *WATCHDOG TEST*\n\n"
            "Dies ist eine Test\\-Nachricht vom KI Trading Watchdog\\.\n\n"
            "✅ Telegram\\-Integration funktioniert\\!"
        )

        async with httpx.AsyncClient(timeout=10) as client:
            for chat_id in self.chat_ids:
                try:
                    response = await client.post(
                        f"{self.api_url}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": message,
                            "parse_mode": "MarkdownV2"
                        }
                    )
                    if response.status_code == 200:
                        results["success"].append(chat_id)
                        logger.info(f"Test message sent to {chat_id}")
                    else:
                        error_desc = response.json().get("description", "Unknown error")
                        results["failed"].append({"chat_id": chat_id, "error": error_desc})
                        logger.error(f"Failed to send test to {chat_id}: {error_desc}")
                except Exception as e:
                    results["failed"].append({"chat_id": chat_id, "error": str(e)})
                    logger.error(f"Exception sending test to {chat_id}: {e}")

        return results

    async def _send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Sendet eine Nachricht an alle Chat-IDs."""
        success_count = 0

        async with httpx.AsyncClient(timeout=10) as client:
            for chat_id in self.chat_ids:
                try:
                    response = await client.post(
                        f"{self.api_url}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": message,
                            "parse_mode": parse_mode
                        }
                    )
                    if response.status_code == 200:
                        logger.info(f"Telegram message sent to {chat_id}")
                        success_count += 1
                    else:
                        logger.error(f"Telegram API error for {chat_id}: {response.text}")
                except Exception as e:
                    logger.error(f"Failed to send Telegram to {chat_id}: {e}")

        return success_count > 0
