# TradingChart Komponenten

Wiederverwendbare Chart-Komponenten für Trading-Anwendungen basierend auf Lightweight Charts (TradingView).

## Komponenten

### TradingChart

Haupt-Chart-Komponente mit Candlestick-Darstellung und Indikatoren.

### ChartToolbar

Toolbar zur dynamischen Steuerung der Indikatoren.

## Installation

Die Komponenten benötigen die Lightweight Charts Bibliothek:

```html
<!-- Lightweight Charts -->
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>

<!-- Chart-Komponenten -->
<script src="js/trading-chart.js"></script>
<script src="js/chart-toolbar.js"></script>
```

## Verwendung

### Basis-Beispiel

```html
<div id="chart-toolbar"></div>
<div id="chart-container" style="height: 500px;"></div>

<script>
    // Chart erstellen
    const chart = new TradingChart('chart-container', {
        symbol: 'BTCUSD',
        timeframe: 'H1',
        theme: 'dark',
        height: 400,
        showVolume: true,
        showLegend: true
    });

    // Toolbar erstellen
    const toolbar = new ChartToolbar('chart-toolbar', chart, {
        theme: 'dark',
        showGroups: true,
        compact: false
    });

    // OHLC-Daten laden
    async function loadChartData() {
        const response = await fetch('/data/api/v1/ohlc/BTCUSD?timeframe=H1&limit=200');
        const data = await response.json();

        // Daten im Format: { time, open, high, low, close, volume }
        chart.setOHLCData(data.data);
    }

    loadChartData();
</script>
```

### Daten-Format

```javascript
const ohlcData = [
    { time: '2024-01-15T10:00:00Z', open: 42000, high: 42500, low: 41800, close: 42300, volume: 1500 },
    { time: '2024-01-15T11:00:00Z', open: 42300, high: 42800, low: 42200, close: 42600, volume: 1200 },
    // ...
];

chart.setOHLCData(ohlcData);
```

### Indikatoren steuern

```javascript
// Einzelne Indikatoren ein-/ausschalten
chart.toggleIndicator('sma20', true);
chart.toggleIndicator('rsi', true);
chart.toggleIndicator('bb', true);

// Aktuellen Status abfragen
const states = chart.getIndicatorStates();
console.log(states); // { sma20: true, rsi: true, bb: true, ... }
```

### Verfügbare Indikatoren

| Key | Beschreibung | Typ |
|-----|--------------|-----|
| `sma20` | Simple Moving Average (20) | Overlay |
| `sma50` | Simple Moving Average (50) | Overlay |
| `sma200` | Simple Moving Average (200) | Overlay |
| `ema12` | Exponential Moving Average (12) | Overlay |
| `ema26` | Exponential Moving Average (26) | Overlay |
| `bb` | Bollinger Bands (20, 2) | Overlay |
| `rsi` | Relative Strength Index (14) | Sub-Chart |
| `macd` | MACD (12, 26, 9) | Sub-Chart |
| `volume` | Volume | Overlay |

### Entry/Exit Levels

```javascript
// Levels setzen
chart.setLevels({
    entry: 42000,
    stopLoss: 41500,
    takeProfit1: 43000,
    takeProfit2: 44000,
    takeProfit3: 45000
}, 'long');

// Levels entfernen
chart.clearLevels();
```

### Marker setzen

```javascript
chart.setMarkers([
    {
        time: '2024-01-15T14:00:00Z',
        position: 'belowBar',
        color: '#22c55e',
        shape: 'arrowUp',
        text: 'BUY'
    },
    {
        time: '2024-01-16T10:00:00Z',
        position: 'aboveBar',
        color: '#ef4444',
        shape: 'arrowDown',
        text: 'SELL'
    }
]);
```

### Toolbar Events

```javascript
const toolbar = new ChartToolbar('toolbar', chart);

// Event wenn Indikator umgeschaltet wird
document.getElementById('toolbar').addEventListener('indicatorToggle', (e) => {
    console.log(`${e.detail.indicator} ist jetzt ${e.detail.enabled ? 'aktiv' : 'inaktiv'}`);
});

// Event wenn Preset angewendet wird
document.getElementById('toolbar').addEventListener('presetApplied', (e) => {
    console.log(`Preset '${e.detail.preset}' angewendet`);
});
```

### Symbol/Timeframe ändern

```javascript
// Nur Label aktualisieren
chart.updateSymbolTimeframe('EURUSD', 'H4');

// Neue Daten laden
async function changeSymbol(symbol, timeframe) {
    chart.updateSymbolTimeframe(symbol, timeframe);

    const response = await fetch(`/data/api/v1/ohlc/${symbol}?timeframe=${timeframe}&limit=200`);
    const data = await response.json();
    chart.setOHLCData(data.data);
}
```

### Theme wechseln

```javascript
// Bei Erstellung
const chart = new TradingChart('container', {
    theme: 'light' // oder 'dark'
});

// Für Theme-Wechsel muss der Chart neu erstellt werden
chart.destroy();
const newChart = new TradingChart('container', { theme: 'light' });
```

### Aufräumen

```javascript
// Chart und Toolbar zerstören
chart.destroy();
toolbar.destroy();
```

## Integration in workplace.html

```html
<!-- Im <head> Bereich -->
<script src="js/trading-chart.js"></script>
<script src="js/chart-toolbar.js"></script>

<!-- Im Detail-Modal -->
<div class="detail-chart-section">
    <div id="detail-chart-toolbar"></div>
    <div id="detail-chart-container" style="height: 400px;"></div>
</div>

<script>
let detailChart = null;
let detailToolbar = null;

async function showDetailChart(symbol, timeframe, levels) {
    // Bestehenden Chart entfernen
    if (detailChart) {
        detailChart.destroy();
        detailToolbar.destroy();
    }

    // Neuen Chart erstellen
    detailChart = new TradingChart('detail-chart-container', {
        symbol: symbol,
        timeframe: timeframe,
        theme: 'dark',
        height: 350
    });

    detailToolbar = new ChartToolbar('detail-chart-toolbar', detailChart, {
        theme: 'dark',
        compact: true
    });

    // OHLC-Daten laden
    try {
        const response = await fetch(`/data/api/v1/ohlc/${symbol}?timeframe=${timeframe}&limit=100`);
        const data = await response.json();

        if (data.data && data.data.length > 0) {
            detailChart.setOHLCData(data.data);
        }
    } catch (error) {
        console.error('Fehler beim Laden der Chart-Daten:', error);
    }

    // Levels setzen falls vorhanden
    if (levels && levels.entry_price) {
        detailChart.setLevels({
            entry: levels.entry_price,
            stopLoss: levels.stop_loss,
            takeProfit1: levels.take_profit_1,
            takeProfit2: levels.take_profit_2,
            takeProfit3: levels.take_profit_3
        }, levels.direction || 'long');
    }

    // Standard-Indikatoren aktivieren
    detailChart.toggleIndicator('sma20', true);
    detailChart.toggleIndicator('volume', true);
}
</script>
```

## API Referenz

### TradingChart

#### Konstruktor

```javascript
new TradingChart(containerId, options)
```

| Option | Typ | Default | Beschreibung |
|--------|-----|---------|--------------|
| `symbol` | string | 'SYMBOL' | Trading-Symbol |
| `timeframe` | string | 'H1' | Timeframe |
| `theme` | string | 'dark' | 'dark' oder 'light' |
| `height` | number | 400 | Chart-Höhe in Pixel |
| `showVolume` | boolean | true | Volume anzeigen |
| `showLegend` | boolean | true | Legende anzeigen |

#### Methoden

| Methode | Beschreibung |
|---------|--------------|
| `setOHLCData(data)` | Setzt OHLC-Daten |
| `toggleIndicator(key, enabled)` | Schaltet Indikator ein/aus |
| `setLevels(levels, direction)` | Setzt Entry/Exit Levels |
| `clearLevels()` | Entfernt alle Levels |
| `setMarkers(markers)` | Setzt Marker |
| `clearMarkers()` | Entfernt alle Marker |
| `getIndicatorStates()` | Gibt Indikator-Status zurück |
| `updateSymbolTimeframe(symbol, tf)` | Aktualisiert Labels |
| `destroy()` | Zerstört Chart-Instanz |

### ChartToolbar

#### Konstruktor

```javascript
new ChartToolbar(containerId, chart, options)
```

| Option | Typ | Default | Beschreibung |
|--------|-----|---------|--------------|
| `position` | string | 'top' | Position der Toolbar |
| `compact` | boolean | false | Kompakter Modus |
| `theme` | string | 'dark' | 'dark' oder 'light' |
| `showGroups` | boolean | true | Indikatoren gruppieren |

#### Methoden

| Methode | Beschreibung |
|---------|--------------|
| `setIndicatorState(key, enabled)` | Setzt Button-Status |
| `getActiveIndicators()` | Gibt aktive Indikatoren zurück |
| `destroy()` | Zerstört Toolbar |

#### Events

| Event | Detail | Beschreibung |
|-------|--------|--------------|
| `indicatorToggle` | `{ indicator, enabled }` | Indikator umgeschaltet |
| `presetApplied` | `{ preset, settings }` | Preset angewendet |
