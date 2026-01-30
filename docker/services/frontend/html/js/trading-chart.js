/**
 * TradingChart - Wiederverwendbare Chart-Komponente für Trading-Anwendungen
 *
 * Basiert auf Lightweight Charts (TradingView)
 *
 * Features:
 * - Candlestick-Chart mit OHLC-Daten
 * - Overlay-Indikatoren (SMA, EMA, Bollinger Bands)
 * - Sub-Chart-Indikatoren (RSI, MACD, Volume)
 * - Entry/Exit-Level Marker
 * - Dynamisches Ein-/Ausblenden von Indikatoren
 *
 * @example
 * const chart = new TradingChart('chart-container', {
 *     symbol: 'BTCUSD',
 *     timeframe: 'H1',
 *     theme: 'dark'
 * });
 *
 * // Daten laden
 * chart.setOHLCData(ohlcData);
 *
 * // Indikatoren ein-/ausblenden
 * chart.toggleIndicator('sma20', true);
 * chart.toggleIndicator('rsi', true);
 *
 * // Entry/Exit Levels setzen
 * chart.setLevels({ entry: 100, stopLoss: 95, takeProfit1: 110 });
 */

class TradingChart {
    /**
     * Erstellt eine neue TradingChart-Instanz
     * @param {string} containerId - ID des Container-Elements
     * @param {Object} options - Konfigurationsoptionen
     */
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);

        if (!this.container) {
            throw new Error(`Container mit ID '${containerId}' nicht gefunden`);
        }

        // Standard-Optionen
        this.options = {
            symbol: options.symbol || 'SYMBOL',
            timeframe: options.timeframe || 'H1',
            theme: options.theme || 'dark',
            height: options.height || 400,
            showVolume: options.showVolume !== false,
            showLegend: options.showLegend !== false,
            ...options
        };

        // Theme-Konfigurationen
        this.themes = {
            dark: {
                background: '#131722',
                text: '#d1d4dc',
                grid: 'rgba(255, 255, 255, 0.05)',
                border: 'rgba(255, 255, 255, 0.1)',
                upColor: '#26a69a',
                downColor: '#ef5350',
                volumeUp: 'rgba(38, 166, 154, 0.5)',
                volumeDown: 'rgba(239, 83, 80, 0.5)',
            },
            light: {
                background: '#ffffff',
                text: '#131722',
                grid: 'rgba(0, 0, 0, 0.05)',
                border: 'rgba(0, 0, 0, 0.1)',
                upColor: '#26a69a',
                downColor: '#ef5350',
                volumeUp: 'rgba(38, 166, 154, 0.5)',
                volumeDown: 'rgba(239, 83, 80, 0.5)',
            }
        };

        // Aktives Theme
        this.theme = this.themes[this.options.theme] || this.themes.dark;

        // Chart-Instanzen
        this.mainChart = null;
        this.rsiChart = null;
        this.macdChart = null;

        // Serien-Referenzen
        this.series = {
            candlestick: null,
            volume: null,
            // Overlay-Indikatoren
            sma20: null,
            sma50: null,
            sma200: null,
            ema12: null,
            ema26: null,
            bbUpper: null,
            bbMiddle: null,
            bbLower: null,
            // Sub-Chart-Indikatoren
            rsi: null,
            macdLine: null,
            signalLine: null,
            macdHistogram: null,
        };

        // Marker-Referenzen
        this.markers = [];
        this.priceLines = [];

        // Indikator-Status
        this.indicatorStates = {
            sma20: false,
            sma50: false,
            sma200: false,
            ema12: false,
            ema26: false,
            bb: false,
            rsi: false,
            macd: false,
            volume: this.options.showVolume,
        };

        // OHLC-Daten Cache
        this.ohlcData = [];

        // ResizeObserver
        this.resizeObserver = null;

        // Initialisieren
        this._init();
    }

    /**
     * Initialisiert die Chart-Komponente
     * @private
     */
    _init() {
        // Container vorbereiten
        this.container.innerHTML = '';
        this.container.style.position = 'relative';

        // Wrapper erstellen
        this._createWrapper();

        // Main Chart erstellen
        this._createMainChart();

        // Legend erstellen (falls aktiviert)
        if (this.options.showLegend) {
            this._createLegend();
        }

        // ResizeObserver einrichten
        this._setupResizeObserver();
    }

    /**
     * Erstellt den Wrapper für Charts und Controls
     * @private
     */
    _createWrapper() {
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'trading-chart-wrapper';
        this.wrapper.style.cssText = `
            display: flex;
            flex-direction: column;
            width: 100%;
            height: 100%;
            background: ${this.theme.background};
            border-radius: 8px;
            overflow: hidden;
        `;
        this.container.appendChild(this.wrapper);

        // Main Chart Container
        this.mainChartContainer = document.createElement('div');
        this.mainChartContainer.className = 'trading-chart-main';
        this.mainChartContainer.style.cssText = `
            flex: 1;
            min-height: ${this.options.height}px;
        `;
        this.wrapper.appendChild(this.mainChartContainer);

        // RSI Chart Container (initially hidden)
        this.rsiChartContainer = document.createElement('div');
        this.rsiChartContainer.className = 'trading-chart-rsi';
        this.rsiChartContainer.style.cssText = `
            height: 100px;
            display: none;
            border-top: 1px solid ${this.theme.border};
        `;
        this.wrapper.appendChild(this.rsiChartContainer);

        // MACD Chart Container (initially hidden)
        this.macdChartContainer = document.createElement('div');
        this.macdChartContainer.className = 'trading-chart-macd';
        this.macdChartContainer.style.cssText = `
            height: 100px;
            display: none;
            border-top: 1px solid ${this.theme.border};
        `;
        this.wrapper.appendChild(this.macdChartContainer);
    }

    /**
     * Erstellt den Haupt-Chart
     * @private
     */
    _createMainChart() {
        this.mainChart = LightweightCharts.createChart(this.mainChartContainer, {
            width: this.mainChartContainer.clientWidth,
            height: this.options.height,
            layout: {
                background: { type: 'solid', color: this.theme.background },
                textColor: this.theme.text,
            },
            grid: {
                vertLines: { color: this.theme.grid },
                horzLines: { color: this.theme.grid },
            },
            rightPriceScale: {
                borderColor: this.theme.border,
                scaleMargins: { top: 0.1, bottom: 0.2 },
            },
            timeScale: {
                borderColor: this.theme.border,
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 5,  // Abstand zum rechten Rand (Anzahl Kerzen)
                minBarSpacing: 3,
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {
                    color: 'rgba(255, 255, 255, 0.4)',
                    width: 1,
                    style: LightweightCharts.LineStyle.Dashed,
                },
                horzLine: {
                    color: 'rgba(255, 255, 255, 0.4)',
                    width: 1,
                    style: LightweightCharts.LineStyle.Dashed,
                },
            },
        });

        // Candlestick-Serie
        this.series.candlestick = this.mainChart.addCandlestickSeries({
            upColor: this.theme.upColor,
            downColor: this.theme.downColor,
            borderUpColor: this.theme.upColor,
            borderDownColor: this.theme.downColor,
            wickUpColor: this.theme.upColor,
            wickDownColor: this.theme.downColor,
        });

        // Volume-Serie (auf separater Skala)
        if (this.options.showVolume) {
            this.series.volume = this.mainChart.addHistogramSeries({
                priceFormat: { type: 'volume' },
                priceScaleId: 'volume',
            });

            this.mainChart.priceScale('volume').applyOptions({
                scaleMargins: { top: 0.85, bottom: 0 },
            });
        }
    }

    /**
     * Erstellt den RSI-Chart
     * @private
     */
    _createRSIChart() {
        if (this.rsiChart) return;

        this.rsiChartContainer.style.display = 'block';

        this.rsiChart = LightweightCharts.createChart(this.rsiChartContainer, {
            width: this.rsiChartContainer.clientWidth,
            height: 100,
            layout: {
                background: { type: 'solid', color: this.theme.background },
                textColor: this.theme.text,
            },
            grid: {
                vertLines: { color: this.theme.grid },
                horzLines: { color: this.theme.grid },
            },
            rightPriceScale: {
                borderColor: this.theme.border,
                scaleMargins: { top: 0.1, bottom: 0.1 },
            },
            timeScale: {
                visible: false,
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
        });

        // RSI Serie
        this.series.rsi = this.rsiChart.addLineSeries({
            color: '#f59e0b',
            lineWidth: 2,
            title: 'RSI(14)',
        });

        // Overbought/Oversold Linien
        const rsiOverbought = this.rsiChart.addLineSeries({
            color: 'rgba(239, 68, 68, 0.5)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
        });
        const rsiOversold = this.rsiChart.addLineSeries({
            color: 'rgba(34, 197, 94, 0.5)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
        });

        // Sync time scales
        this.mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
            if (range && this.rsiChart) {
                this.rsiChart.timeScale().setVisibleLogicalRange(range);
            }
        });

        return { overbought: rsiOverbought, oversold: rsiOversold };
    }

    /**
     * Erstellt den MACD-Chart
     * @private
     */
    _createMACDChart() {
        if (this.macdChart) return;

        this.macdChartContainer.style.display = 'block';

        this.macdChart = LightweightCharts.createChart(this.macdChartContainer, {
            width: this.macdChartContainer.clientWidth,
            height: 100,
            layout: {
                background: { type: 'solid', color: this.theme.background },
                textColor: this.theme.text,
            },
            grid: {
                vertLines: { color: this.theme.grid },
                horzLines: { color: this.theme.grid },
            },
            rightPriceScale: {
                borderColor: this.theme.border,
            },
            timeScale: {
                visible: false,
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
        });

        // MACD Line
        this.series.macdLine = this.macdChart.addLineSeries({
            color: '#3b82f6',
            lineWidth: 2,
            title: 'MACD',
        });

        // Signal Line
        this.series.signalLine = this.macdChart.addLineSeries({
            color: '#f59e0b',
            lineWidth: 2,
            title: 'Signal',
        });

        // MACD Histogram
        this.series.macdHistogram = this.macdChart.addHistogramSeries({
            color: '#26a69a',
        });

        // Sync time scales
        this.mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
            if (range && this.macdChart) {
                this.macdChart.timeScale().setVisibleLogicalRange(range);
            }
        });
    }

    /**
     * Erstellt die Legende
     * @private
     */
    _createLegend() {
        this.legend = document.createElement('div');
        this.legend.className = 'trading-chart-legend';
        this.legend.style.cssText = `
            position: absolute;
            top: 8px;
            left: 8px;
            z-index: 10;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            font-size: 11px;
            color: ${this.theme.text};
            pointer-events: none;
        `;
        this.mainChartContainer.style.position = 'relative';
        this.mainChartContainer.appendChild(this.legend);

        this._updateLegend();
    }

    /**
     * Aktualisiert die Legende
     * @private
     */
    _updateLegend() {
        if (!this.legend) return;

        const items = [
            `<span style="font-weight: 600;">${this.options.symbol}</span>`,
            `<span style="opacity: 0.7;">${this.options.timeframe}</span>`,
        ];

        // Aktive Indikatoren anzeigen
        if (this.indicatorStates.sma20) items.push('<span style="color: #f59e0b;">SMA20</span>');
        if (this.indicatorStates.sma50) items.push('<span style="color: #8b5cf6;">SMA50</span>');
        if (this.indicatorStates.sma200) items.push('<span style="color: #ec4899;">SMA200</span>');
        if (this.indicatorStates.ema12) items.push('<span style="color: #10b981;">EMA12</span>');
        if (this.indicatorStates.ema26) items.push('<span style="color: #ef4444;">EMA26</span>');
        if (this.indicatorStates.bb) items.push('<span style="color: #64c8ff;">BB</span>');

        this.legend.innerHTML = items.join(' | ');
    }

    /**
     * Richtet den ResizeObserver ein
     * @private
     */
    _setupResizeObserver() {
        this.resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width } = entry.contentRect;
                if (this.mainChart) {
                    this.mainChart.applyOptions({ width });
                }
                if (this.rsiChart) {
                    this.rsiChart.applyOptions({ width });
                }
                if (this.macdChart) {
                    this.macdChart.applyOptions({ width });
                }
            }
        });
        this.resizeObserver.observe(this.mainChartContainer);
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================

    /**
     * Setzt OHLC-Daten
     * @param {Array} data - Array von OHLC-Objekten { time, open, high, low, close, volume? }
     */
    setOHLCData(data) {
        if (!data || !Array.isArray(data)) {
            console.warn('TradingChart: Ungültige OHLC-Daten');
            return;
        }

        this.ohlcData = data;

        // Candlestick-Daten setzen
        const candleData = data.map(d => ({
            time: this._normalizeTime(d.time),
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        this.series.candlestick.setData(candleData);

        // Volume-Daten setzen
        if (this.series.volume && this.indicatorStates.volume) {
            const volumeData = data.map(d => ({
                time: this._normalizeTime(d.time),
                value: d.volume || 0,
                color: d.close >= d.open ? this.theme.volumeUp : this.theme.volumeDown,
            }));
            this.series.volume.setData(volumeData);
        }

        // Aktive Indikatoren neu berechnen
        this._recalculateIndicators();

        // Chart anpassen
        this.mainChart.timeScale().fitContent();
    }

    /**
     * Schaltet einen Indikator ein/aus
     * @param {string} indicator - Name des Indikators
     * @param {boolean} enabled - Ein/Aus
     */
    toggleIndicator(indicator, enabled) {
        this.indicatorStates[indicator] = enabled;

        switch (indicator) {
            case 'sma20':
            case 'sma50':
            case 'sma200':
                this._toggleSMA(indicator, enabled);
                break;
            case 'ema12':
            case 'ema26':
                this._toggleEMA(indicator, enabled);
                break;
            case 'bb':
                this._toggleBollingerBands(enabled);
                break;
            case 'rsi':
                this._toggleRSI(enabled);
                break;
            case 'macd':
                this._toggleMACD(enabled);
                break;
            case 'volume':
                this._toggleVolume(enabled);
                break;
        }

        this._updateLegend();
    }

    /**
     * Setzt Entry/Exit-Levels
     * @param {Object} levels - { entry, stopLoss, takeProfit1, takeProfit2, takeProfit3 }
     * @param {string} direction - 'long' oder 'short'
     */
    setLevels(levels, direction = 'long') {
        // Bestehende Lines entfernen
        this.clearLevels();

        const isLong = direction === 'long';

        if (levels.entry) {
            this._addPriceLine(levels.entry, 'Entry', '#3b82f6', LightweightCharts.LineStyle.Solid);
        }

        if (levels.stopLoss) {
            this._addPriceLine(levels.stopLoss, 'SL', '#ef4444', LightweightCharts.LineStyle.Dashed);
        }

        if (levels.takeProfit1) {
            this._addPriceLine(levels.takeProfit1, 'TP1', '#22c55e', LightweightCharts.LineStyle.Dashed);
        }

        if (levels.takeProfit2) {
            this._addPriceLine(levels.takeProfit2, 'TP2', '#22c55e', LightweightCharts.LineStyle.Dotted);
        }

        if (levels.takeProfit3) {
            this._addPriceLine(levels.takeProfit3, 'TP3', '#22c55e', LightweightCharts.LineStyle.Dotted);
        }
    }

    /**
     * Entfernt alle Level-Linien
     */
    clearLevels() {
        this.priceLines.forEach(line => {
            this.series.candlestick.removePriceLine(line);
        });
        this.priceLines = [];
    }

    /**
     * Fügt Marker hinzu (Entry, Exit, Signale)
     * @param {Array} markers - Array von { time, position, color, shape, text }
     */
    setMarkers(markers) {
        if (!markers || !Array.isArray(markers)) return;

        const formattedMarkers = markers.map(m => ({
            time: this._normalizeTime(m.time),
            position: m.position || 'belowBar',
            color: m.color || '#3b82f6',
            shape: m.shape || 'arrowUp',
            text: m.text || '',
        }));

        this.series.candlestick.setMarkers(formattedMarkers);
        this.markers = formattedMarkers;
    }

    /**
     * Entfernt alle Marker
     */
    clearMarkers() {
        this.series.candlestick.setMarkers([]);
        this.markers = [];
    }

    /**
     * Gibt den aktuellen Indikator-Status zurück
     * @returns {Object} Status aller Indikatoren
     */
    getIndicatorStates() {
        return { ...this.indicatorStates };
    }

    /**
     * Aktualisiert Symbol und Timeframe
     * @param {string} symbol
     * @param {string} timeframe
     */
    updateSymbolTimeframe(symbol, timeframe) {
        this.options.symbol = symbol;
        this.options.timeframe = timeframe;
        this._updateLegend();
    }

    /**
     * Zerstört die Chart-Instanz und räumt auf
     */
    destroy() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        if (this.mainChart) {
            this.mainChart.remove();
        }
        if (this.rsiChart) {
            this.rsiChart.remove();
        }
        if (this.macdChart) {
            this.macdChart.remove();
        }
        this.container.innerHTML = '';
    }

    // =========================================================================
    // PRIVATE INDICATOR METHODS
    // =========================================================================

    /**
     * Normalisiert Zeitstempel
     * @private
     */
    _normalizeTime(time) {
        if (typeof time === 'number') {
            return time;
        }
        if (typeof time === 'string') {
            return Math.floor(new Date(time).getTime() / 1000);
        }
        if (time instanceof Date) {
            return Math.floor(time.getTime() / 1000);
        }
        return time;
    }

    /**
     * Fügt eine Preis-Linie hinzu
     * @private
     */
    _addPriceLine(price, title, color, lineStyle) {
        const line = this.series.candlestick.createPriceLine({
            price: price,
            color: color,
            lineWidth: 1,
            lineStyle: lineStyle,
            axisLabelVisible: true,
            title: title,
        });
        this.priceLines.push(line);
    }

    /**
     * Berechnet alle aktiven Indikatoren neu
     * @private
     */
    _recalculateIndicators() {
        if (this.indicatorStates.sma20) this._calculateSMA('sma20', 20);
        if (this.indicatorStates.sma50) this._calculateSMA('sma50', 50);
        if (this.indicatorStates.sma200) this._calculateSMA('sma200', 200);
        if (this.indicatorStates.ema12) this._calculateEMA('ema12', 12);
        if (this.indicatorStates.ema26) this._calculateEMA('ema26', 26);
        if (this.indicatorStates.bb) this._calculateBollingerBands();
        if (this.indicatorStates.rsi) this._calculateRSI();
        if (this.indicatorStates.macd) this._calculateMACD();
    }

    /**
     * Schaltet SMA ein/aus
     * @private
     */
    _toggleSMA(key, enabled) {
        const colors = {
            sma20: '#f59e0b',
            sma50: '#8b5cf6',
            sma200: '#ec4899',
        };
        const periods = { sma20: 20, sma50: 50, sma200: 200 };

        if (enabled) {
            if (!this.series[key]) {
                this.series[key] = this.mainChart.addLineSeries({
                    color: colors[key],
                    lineWidth: 1,
                    title: `SMA${periods[key]}`,
                });
            }
            this._calculateSMA(key, periods[key]);
        } else {
            if (this.series[key]) {
                this.mainChart.removeSeries(this.series[key]);
                this.series[key] = null;
            }
        }
    }

    /**
     * Berechnet SMA
     * @private
     */
    _calculateSMA(key, period) {
        if (!this.series[key] || this.ohlcData.length < period) return;

        const smaData = [];
        for (let i = period - 1; i < this.ohlcData.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += this.ohlcData[i - j].close;
            }
            smaData.push({
                time: this._normalizeTime(this.ohlcData[i].time),
                value: sum / period,
            });
        }
        this.series[key].setData(smaData);
    }

    /**
     * Schaltet EMA ein/aus
     * @private
     */
    _toggleEMA(key, enabled) {
        const colors = { ema12: '#10b981', ema26: '#ef4444' };
        const periods = { ema12: 12, ema26: 26 };

        if (enabled) {
            if (!this.series[key]) {
                this.series[key] = this.mainChart.addLineSeries({
                    color: colors[key],
                    lineWidth: 1,
                    title: `EMA${periods[key]}`,
                });
            }
            this._calculateEMA(key, periods[key]);
        } else {
            if (this.series[key]) {
                this.mainChart.removeSeries(this.series[key]);
                this.series[key] = null;
            }
        }
    }

    /**
     * Berechnet EMA
     * @private
     */
    _calculateEMA(key, period) {
        if (!this.series[key] || this.ohlcData.length < period) return;

        const multiplier = 2 / (period + 1);
        const emaData = [];

        // Erstes EMA ist SMA
        let sum = 0;
        for (let i = 0; i < period; i++) {
            sum += this.ohlcData[i].close;
        }
        let ema = sum / period;

        for (let i = period - 1; i < this.ohlcData.length; i++) {
            if (i === period - 1) {
                emaData.push({
                    time: this._normalizeTime(this.ohlcData[i].time),
                    value: ema,
                });
            } else {
                ema = (this.ohlcData[i].close - ema) * multiplier + ema;
                emaData.push({
                    time: this._normalizeTime(this.ohlcData[i].time),
                    value: ema,
                });
            }
        }
        this.series[key].setData(emaData);
    }

    /**
     * Schaltet Bollinger Bands ein/aus
     * @private
     */
    _toggleBollingerBands(enabled) {
        if (enabled) {
            if (!this.series.bbUpper) {
                this.series.bbUpper = this.mainChart.addLineSeries({
                    color: 'rgba(100, 200, 255, 0.5)',
                    lineWidth: 1,
                });
            }
            if (!this.series.bbMiddle) {
                this.series.bbMiddle = this.mainChart.addLineSeries({
                    color: '#64c8ff',
                    lineWidth: 1,
                });
            }
            if (!this.series.bbLower) {
                this.series.bbLower = this.mainChart.addLineSeries({
                    color: 'rgba(100, 200, 255, 0.5)',
                    lineWidth: 1,
                });
            }
            this._calculateBollingerBands();
        } else {
            ['bbUpper', 'bbMiddle', 'bbLower'].forEach(key => {
                if (this.series[key]) {
                    this.mainChart.removeSeries(this.series[key]);
                    this.series[key] = null;
                }
            });
        }
    }

    /**
     * Berechnet Bollinger Bands
     * @private
     */
    _calculateBollingerBands(period = 20, stdDev = 2) {
        if (this.ohlcData.length < period) return;

        const upperData = [], middleData = [], lowerData = [];

        for (let i = period - 1; i < this.ohlcData.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += this.ohlcData[i - j].close;
            }
            const sma = sum / period;

            let variance = 0;
            for (let j = 0; j < period; j++) {
                variance += Math.pow(this.ohlcData[i - j].close - sma, 2);
            }
            const std = Math.sqrt(variance / period);

            const time = this._normalizeTime(this.ohlcData[i].time);
            upperData.push({ time, value: sma + stdDev * std });
            middleData.push({ time, value: sma });
            lowerData.push({ time, value: sma - stdDev * std });
        }

        if (this.series.bbUpper) this.series.bbUpper.setData(upperData);
        if (this.series.bbMiddle) this.series.bbMiddle.setData(middleData);
        if (this.series.bbLower) this.series.bbLower.setData(lowerData);
    }

    /**
     * Schaltet RSI ein/aus
     * @private
     */
    _toggleRSI(enabled) {
        if (enabled) {
            const rsiLines = this._createRSIChart();
            this._calculateRSI(rsiLines);
        } else {
            if (this.rsiChart) {
                this.rsiChart.remove();
                this.rsiChart = null;
                this.series.rsi = null;
            }
            this.rsiChartContainer.style.display = 'none';
        }
    }

    /**
     * Berechnet RSI
     * @private
     */
    _calculateRSI(rsiLines = null, period = 14) {
        if (this.ohlcData.length < period + 1) return;

        const rsiData = [];
        const overboughtData = [];
        const oversoldData = [];

        let gains = 0, losses = 0;

        // Initiale Gewinne/Verluste berechnen
        for (let i = 1; i <= period; i++) {
            const change = this.ohlcData[i].close - this.ohlcData[i - 1].close;
            if (change > 0) gains += change;
            else losses -= change;
        }

        let avgGain = gains / period;
        let avgLoss = losses / period;

        for (let i = period; i < this.ohlcData.length; i++) {
            const time = this._normalizeTime(this.ohlcData[i].time);

            if (i > period) {
                const change = this.ohlcData[i].close - this.ohlcData[i - 1].close;
                avgGain = (avgGain * (period - 1) + (change > 0 ? change : 0)) / period;
                avgLoss = (avgLoss * (period - 1) + (change < 0 ? -change : 0)) / period;
            }

            const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
            const rsi = 100 - (100 / (1 + rs));

            rsiData.push({ time, value: rsi });
            overboughtData.push({ time, value: 70 });
            oversoldData.push({ time, value: 30 });
        }

        if (this.series.rsi) this.series.rsi.setData(rsiData);
        if (rsiLines) {
            rsiLines.overbought.setData(overboughtData);
            rsiLines.oversold.setData(oversoldData);
        }
    }

    /**
     * Schaltet MACD ein/aus
     * @private
     */
    _toggleMACD(enabled) {
        if (enabled) {
            this._createMACDChart();
            this._calculateMACD();
        } else {
            if (this.macdChart) {
                this.macdChart.remove();
                this.macdChart = null;
                this.series.macdLine = null;
                this.series.signalLine = null;
                this.series.macdHistogram = null;
            }
            this.macdChartContainer.style.display = 'none';
        }
    }

    /**
     * Berechnet MACD
     * @private
     */
    _calculateMACD(fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
        if (this.ohlcData.length < slowPeriod + signalPeriod) return;

        // EMA berechnen
        const calculateEMA = (data, period, startIndex = 0) => {
            const multiplier = 2 / (period + 1);
            let sum = 0;
            for (let i = startIndex; i < startIndex + period; i++) {
                sum += data[i];
            }
            let ema = sum / period;
            const result = new Array(startIndex + period - 1).fill(null);
            result.push(ema);

            for (let i = startIndex + period; i < data.length; i++) {
                ema = (data[i] - ema) * multiplier + ema;
                result.push(ema);
            }
            return result;
        };

        const closes = this.ohlcData.map(d => d.close);
        const ema12 = calculateEMA(closes, fastPeriod);
        const ema26 = calculateEMA(closes, slowPeriod);

        const macdLine = [];
        for (let i = 0; i < closes.length; i++) {
            if (ema12[i] !== null && ema26[i] !== null) {
                macdLine.push(ema12[i] - ema26[i]);
            } else {
                macdLine.push(null);
            }
        }

        const signalLine = calculateEMA(macdLine.filter(v => v !== null), signalPeriod, slowPeriod - 1);

        const macdData = [], signalData = [], histogramData = [];

        let signalIndex = 0;
        for (let i = slowPeriod - 1; i < this.ohlcData.length; i++) {
            const time = this._normalizeTime(this.ohlcData[i].time);
            const macd = macdLine[i];

            if (macd !== null) {
                macdData.push({ time, value: macd });

                if (signalIndex < signalLine.length && signalLine[signalIndex] !== null) {
                    const signal = signalLine[signalIndex];
                    signalData.push({ time, value: signal });

                    const histogram = macd - signal;
                    histogramData.push({
                        time,
                        value: histogram,
                        color: histogram >= 0 ? 'rgba(38, 166, 154, 0.7)' : 'rgba(239, 83, 80, 0.7)',
                    });
                }
                signalIndex++;
            }
        }

        if (this.series.macdLine) this.series.macdLine.setData(macdData);
        if (this.series.signalLine) this.series.signalLine.setData(signalData);
        if (this.series.macdHistogram) this.series.macdHistogram.setData(histogramData);
    }

    /**
     * Schaltet Volume ein/aus
     * @private
     */
    _toggleVolume(enabled) {
        if (enabled) {
            if (!this.series.volume) {
                this.series.volume = this.mainChart.addHistogramSeries({
                    priceFormat: { type: 'volume' },
                    priceScaleId: 'volume',
                });
                this.mainChart.priceScale('volume').applyOptions({
                    scaleMargins: { top: 0.85, bottom: 0 },
                });
            }
            // Volume-Daten setzen
            if (this.ohlcData.length > 0) {
                const volumeData = this.ohlcData.map(d => ({
                    time: this._normalizeTime(d.time),
                    value: d.volume || 0,
                    color: d.close >= d.open ? this.theme.volumeUp : this.theme.volumeDown,
                }));
                this.series.volume.setData(volumeData);
            }
        } else {
            if (this.series.volume) {
                this.mainChart.removeSeries(this.series.volume);
                this.series.volume = null;
            }
        }
    }
}

// Export für Module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TradingChart;
}
