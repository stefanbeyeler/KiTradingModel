[https://github.com/twelvedata/twelvedata-python](https://github.com/twelvedata/twelvedata-python)

f7f2fea53236423086fab90d72d11967

## TwelveData Pläne für Symbol-Datenabfragen

Basierend auf den verfügbaren Informationen, hier eine Übersicht:

### Planübersicht

| Plan | Preis/Monat | API Credits | Märkte |
| --- | --- | --- | --- |
| **Basic** | Kostenlos | 800/Tag | 3 (US, Forex, Crypto) |
| **Grow** | $79 | 377 Credits | 27 Märkte |
| **Pro** | $229 | 1.597 Credits | 47 Märkte |
| **Ultra** | $999 | 10.946 Credits | 83 Märkte |
| **Enterprise** | $1.999 | 17.711+ Credits | Alle 83 Märkte |

### Welchen Plan benötigen Sie?

Das hängt von Ihrem Anwendungsfall ab:

**Basic (Kostenlos)** reicht wenn:

*   Sie nur **US-Aktien, Forex oder Crypto** abfragen
*   Max. 800 API-Calls pro Tag ausreichen
*   Keine Echtzeit-WebSocket-Daten benötigt werden

**Grow ($79/Monat)** wenn:

*   Sie Zugang zu **24 zusätzlichen Märkten** benötigen
*   Sie **Commodities-Daten** brauchen
*   Keine täglichen Limits gewünscht sind

**Pro ($229/Monat)** wenn:

*   **Pre/Post-Market US-Daten** benötigt werden
*   **Batch-Requests** für effizientere Abfragen
*   **Market Movers** Daten

**Ultra/Enterprise** für:

*   **Fundamentaldaten** (Bilanzen, Gewinn-/Verlustrechnung)
*   Hohe API-Volumina
*   Alle globalen Märkte

### Credit-System

Beachten Sie: **1 Standard-Abfrage = 1 Credit pro Symbol**, aber komplexere Daten wie Fundamentaldaten kosten **100 Credits pro Symbol**.

**Empfehlung für KiTradingModel:** Für grundlegende Kurs- und Indikator-Daten von Crypto/Forex reicht der **Basic-Plan** zum Testen. Für Produktivbetrieb mit höherem Volumen wäre **Grow** oder **Pro** sinnvoll.