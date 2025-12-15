"""Economic Calendar Data Source - Fed, ECB, BOJ decisions, NFP, CPI, GDP etc."""

import aiohttp
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from .base import DataSourceBase, DataSourceResult, DataSourceType, DataPriority


class EconomicCalendarSource(DataSourceBase):
    """
    Fetches economic calendar events from various sources.

    Data includes:
    - Central Bank decisions (Fed, ECB, BOJ, BOE, SNB)
    - Employment data (NFP, Jobless Claims, Unemployment Rate)
    - Inflation data (CPI, PPI, PCE)
    - GDP and economic growth indicators
    - Retail Sales, PMI, ISM
    - Earnings calendar for major companies
    """

    source_type = DataSourceType.ECONOMIC_CALENDAR

    # High-impact events that should be prioritized
    HIGH_IMPACT_EVENTS = [
        "interest rate decision", "fed funds rate", "ecb rate",
        "non-farm payroll", "nfp", "unemployment rate",
        "cpi", "consumer price index", "inflation",
        "gdp", "gross domestic product",
        "fomc", "monetary policy", "press conference",
        "pce", "personal consumption"
    ]

    CRITICAL_EVENTS = [
        "fed rate", "fomc decision", "emergency meeting",
        "ecb rate decision", "boj rate"
    ]

    # Symbol to currency/region mapping
    SYMBOL_REGION_MAP = {
        "BTCUSD": ["USD", "global", "crypto"],
        "ETHUSD": ["USD", "global", "crypto"],
        "XAUUSD": ["USD", "global", "gold"],
        "EURUSD": ["EUR", "USD", "eurozone", "us"],
        "GBPUSD": ["GBP", "USD", "uk", "us"],
        "USDJPY": ["USD", "JPY", "us", "japan"],
    }

    def __init__(self):
        super().__init__()
        self._cache_ttl = 1800  # 30 minutes for calendar data

    async def fetch(self, symbol: Optional[str] = None, **kwargs) -> list[DataSourceResult]:
        """
        Fetch economic calendar events.

        Args:
            symbol: Trading symbol to get relevant events for
            days_ahead: Number of days to look ahead (default: 7)
            days_back: Number of days to look back (default: 1)

        Returns:
            List of economic calendar events
        """
        cache_key = self._get_cache_key(symbol, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        days_ahead = kwargs.get("days_ahead", 7)
        days_back = kwargs.get("days_back", 1)

        results = []

        # Fetch from multiple sources for redundancy
        try:
            # Try primary source: Forex Factory style API
            events = await self._fetch_calendar_events(days_ahead, days_back)

            for event in events:
                priority = self._determine_priority(event)
                relevance = self._check_symbol_relevance(event, symbol)

                if not relevance and symbol:
                    continue

                content = self._format_event_content(event)

                result = DataSourceResult(
                    source_type=self.source_type,
                    content=content,
                    symbol=symbol,
                    timestamp=event.get("datetime", datetime.utcnow()),
                    priority=priority,
                    metadata={
                        "event_name": event.get("name", ""),
                        "country": event.get("country", ""),
                        "impact": event.get("impact", "medium"),
                        "actual": event.get("actual"),
                        "forecast": event.get("forecast"),
                        "previous": event.get("previous"),
                        "relevance_score": relevance
                    },
                    raw_data=event
                )
                results.append(result)

        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            # Return fallback data
            results.append(self._create_fallback_result(symbol))

        self._set_cache(cache_key, results)
        return results

    async def fetch_for_rag(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        """Fetch economic events formatted for RAG storage."""
        results = await self.fetch(symbol, **kwargs)
        return [r.to_rag_document() for r in results]

    async def _fetch_calendar_events(
        self,
        days_ahead: int,
        days_back: int
    ) -> list[dict]:
        """
        Fetch calendar events from API or generate from known schedule.

        In production, this would connect to:
        - Forex Factory API
        - Investing.com Calendar
        - TradingEconomics API
        - Alpha Vantage Economic Calendar
        """
        events = []
        now = datetime.utcnow()

        # Generate known recurring events
        events.extend(self._generate_recurring_events(now, days_ahead, days_back))

        # Try to fetch from free APIs
        try:
            api_events = await self._fetch_from_apis()
            events.extend(api_events)
        except Exception as e:
            logger.warning(f"Could not fetch from external APIs: {e}")

        # Deduplicate and sort by date
        seen = set()
        unique_events = []
        for event in events:
            key = (event.get("name", ""), event.get("datetime", now).date())
            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        unique_events.sort(key=lambda x: x.get("datetime", now))
        return unique_events

    def _generate_recurring_events(
        self,
        now: datetime,
        days_ahead: int,
        days_back: int
    ) -> list[dict]:
        """Generate known recurring economic events."""
        events = []
        start_date = now - timedelta(days=days_back)
        end_date = now + timedelta(days=days_ahead)

        # FOMC meetings (typically 8 per year, every 6-7 weeks)
        fomc_dates = self._get_fomc_dates(start_date, end_date)
        for date in fomc_dates:
            events.append({
                "name": "FOMC Interest Rate Decision",
                "datetime": date,
                "country": "US",
                "impact": "high",
                "description": "Federal Reserve monetary policy decision. Key event for USD and global markets.",
                "affects": ["USD", "stocks", "crypto", "bonds", "gold"]
            })

        # First Friday NFP
        current = start_date
        while current <= end_date:
            if current.weekday() == 4:  # Friday
                if current.day <= 7:  # First Friday
                    events.append({
                        "name": "US Non-Farm Payrolls (NFP)",
                        "datetime": current.replace(hour=13, minute=30),
                        "country": "US",
                        "impact": "high",
                        "description": "Monthly employment report. Major market mover for USD and risk assets.",
                        "affects": ["USD", "stocks", "crypto", "gold"]
                    })
            current += timedelta(days=1)

        # Monthly CPI (typically mid-month)
        current = start_date.replace(day=10)
        while current <= end_date:
            events.append({
                "name": "US CPI (Consumer Price Index)",
                "datetime": current.replace(hour=13, minute=30),
                "country": "US",
                "impact": "high",
                "description": "Monthly inflation data. Critical for Fed policy expectations.",
                "affects": ["USD", "bonds", "gold", "stocks", "crypto"]
            })
            # Move to next month's ~12th
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=12)
            else:
                current = current.replace(month=current.month + 1, day=12)

        # ECB Rate Decision (typically every 6 weeks)
        ecb_dates = self._get_ecb_dates(start_date, end_date)
        for date in ecb_dates:
            events.append({
                "name": "ECB Interest Rate Decision",
                "datetime": date,
                "country": "EU",
                "impact": "high",
                "description": "European Central Bank monetary policy decision.",
                "affects": ["EUR", "European stocks", "global bonds"]
            })

        # Weekly Jobless Claims (Thursdays)
        current = start_date
        while current <= end_date:
            if current.weekday() == 3:  # Thursday
                events.append({
                    "name": "US Initial Jobless Claims",
                    "datetime": current.replace(hour=13, minute=30),
                    "country": "US",
                    "impact": "medium",
                    "description": "Weekly unemployment claims data.",
                    "affects": ["USD", "stocks"]
                })
            current += timedelta(days=1)

        return events

    def _get_fomc_dates(self, start: datetime, end: datetime) -> list[datetime]:
        """Get FOMC meeting dates in range (approximate)."""
        # 2024-2025 FOMC schedule (approximate)
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]  # Typical FOMC months
        dates = []

        for year in range(start.year, end.year + 1):
            for month in fomc_months:
                # FOMC typically meets mid-to-late month
                date = datetime(year, month, 18, 19, 0)  # 2 PM EST = 19:00 UTC
                if start <= date <= end:
                    dates.append(date)

        return dates

    def _get_ecb_dates(self, start: datetime, end: datetime) -> list[datetime]:
        """Get ECB meeting dates in range (approximate)."""
        dates = []
        current = start.replace(day=15)  # ECB typically mid-month

        while current <= end:
            dates.append(current.replace(hour=13, minute=15))
            # ECB meets roughly every 6 weeks
            current += timedelta(weeks=6)

        return dates

    async def _fetch_from_apis(self) -> list[dict]:
        """Fetch from free economic calendar APIs."""
        events = []

        # Try Alpha Vantage if API key available
        # try:
        #     async with aiohttp.ClientSession() as session:
        #         async with session.get(url) as response:
        #             data = await response.json()
        #             events.extend(self._parse_alpha_vantage(data))
        # except Exception:
        #     pass

        return events

    def _determine_priority(self, event: dict) -> DataPriority:
        """Determine the priority of an economic event."""
        name_lower = event.get("name", "").lower()
        impact = event.get("impact", "medium").lower()

        # Check for critical events
        for keyword in self.CRITICAL_EVENTS:
            if keyword in name_lower:
                return DataPriority.CRITICAL

        # Check for high impact events
        if impact == "high":
            return DataPriority.HIGH

        for keyword in self.HIGH_IMPACT_EVENTS:
            if keyword in name_lower:
                return DataPriority.HIGH

        if impact == "medium":
            return DataPriority.MEDIUM

        return DataPriority.LOW

    def _check_symbol_relevance(self, event: dict, symbol: Optional[str]) -> float:
        """Check how relevant an event is to a specific symbol (0-1 score)."""
        if not symbol:
            return 1.0

        affects = event.get("affects", [])
        country = event.get("country", "").upper()

        # Get regions relevant to symbol
        relevant_regions = self.SYMBOL_REGION_MAP.get(symbol.upper(), ["USD", "global"])

        score = 0.0

        # Check if event affects relevant currencies/regions
        for region in relevant_regions:
            if region.upper() in [a.upper() for a in affects]:
                score += 0.5
            if region.upper() == country:
                score += 0.3

        # Global events always somewhat relevant
        if "global" in [a.lower() for a in affects]:
            score += 0.2

        # Crypto is affected by USD events
        if symbol.upper() in ["BTCUSD", "ETHUSD", "SOLUSD"] and country == "US":
            score += 0.3

        return min(score, 1.0)

    def _format_event_content(self, event: dict) -> str:
        """Format event data as readable content for RAG."""
        event_time = event.get("datetime", datetime.utcnow())
        time_str = event_time.strftime("%Y-%m-%d %H:%M UTC")

        content = f"""WIRTSCHAFTSKALENDER EVENT
==========================
Event: {event.get('name', 'Unknown')}
Zeit: {time_str}
Land: {event.get('country', 'Global')}
Impact: {event.get('impact', 'medium').upper()}

Beschreibung:
{event.get('description', 'Keine Beschreibung verfügbar.')}

"""

        # Add data if available
        if event.get("actual") is not None:
            content += f"Aktuell: {event['actual']}\n"
        if event.get("forecast") is not None:
            content += f"Prognose: {event['forecast']}\n"
        if event.get("previous") is not None:
            content += f"Vorher: {event['previous']}\n"

        affects = event.get("affects", [])
        if affects:
            content += f"\nBetrifft: {', '.join(affects)}\n"

        # Add trading implications
        content += f"""
Trading-Implikationen:
- Bei höher als erwartet: {self._get_higher_implication(event)}
- Bei niedriger als erwartet: {self._get_lower_implication(event)}
"""

        return content

    def _get_higher_implication(self, event: dict) -> str:
        """Get trading implication if data comes in higher than expected."""
        name = event.get("name", "").lower()

        if "interest rate" in name or "fed" in name or "ecb" in name:
            return "USD/EUR stärker, Aktien/Crypto schwächer (restriktivere Geldpolitik)"
        if "inflation" in name or "cpi" in name:
            return "Höhere Zinserwartungen, USD stärker, Risk-off für Crypto/Aktien"
        if "payroll" in name or "employment" in name:
            return "Starke Wirtschaft, USD stärker, gemischt für Risk Assets"
        if "gdp" in name:
            return "Positiv für Aktien, gemischt für USD"
        if "jobless" in name:
            return "Schwächerer Arbeitsmarkt, dovish Fed Erwartungen"

        return "Marktabhängige Reaktion"

    def _get_lower_implication(self, event: dict) -> str:
        """Get trading implication if data comes in lower than expected."""
        name = event.get("name", "").lower()

        if "interest rate" in name or "fed" in name or "ecb" in name:
            return "USD/EUR schwächer, Aktien/Crypto stärker (lockere Geldpolitik)"
        if "inflation" in name or "cpi" in name:
            return "Niedrigere Zinserwartungen, USD schwächer, Risk-on für Crypto/Aktien"
        if "payroll" in name or "employment" in name:
            return "Schwache Wirtschaft, Fed könnte dovish werden"
        if "gdp" in name:
            return "Negativ für Aktien, mögliche Stimulus-Erwartungen"
        if "jobless" in name:
            return "Stärkerer Arbeitsmarkt, hawkish Fed Erwartungen"

        return "Marktabhängige Reaktion"

    def _create_fallback_result(self, symbol: Optional[str]) -> DataSourceResult:
        """Create fallback result when API fetch fails."""
        return DataSourceResult(
            source_type=self.source_type,
            content="""WIRTSCHAFTSKALENDER - ÜBERSICHT
================================
Hinweis: Live-Kalenderdaten derzeit nicht verfügbar.

Wichtige regelmäßige Events:
- FOMC Entscheidungen: Alle 6-7 Wochen (Mittwoch 19:00 UTC)
- US NFP: Erster Freitag im Monat (13:30 UTC)
- US CPI: Mitte des Monats (13:30 UTC)
- ECB Entscheidung: Alle 6 Wochen (Donnerstag 13:15 UTC)
- Wöchentliche Jobless Claims: Donnerstags (13:30 UTC)

Empfehlung: Vor wichtigen Events Positionen reduzieren oder absichern.
""",
            symbol=symbol,
            priority=DataPriority.MEDIUM,
            metadata={"fallback": True}
        )
