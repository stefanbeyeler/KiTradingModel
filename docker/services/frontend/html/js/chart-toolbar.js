/**
 * ChartToolbar - Wiederverwendbare Toolbar für TradingChart-Indikator-Steuerung
 *
 * @example
 * const toolbar = new ChartToolbar('toolbar-container', tradingChart, {
 *     position: 'top',
 *     compact: false
 * });
 */

class ChartToolbar {
    /**
     * Erstellt eine neue ChartToolbar-Instanz
     * @param {string} containerId - ID des Container-Elements
     * @param {TradingChart} chart - TradingChart-Instanz
     * @param {Object} options - Konfigurationsoptionen
     */
    constructor(containerId, chart, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.chart = chart;

        if (!this.container) {
            throw new Error(`Container mit ID '${containerId}' nicht gefunden`);
        }

        this.options = {
            position: options.position || 'top', // 'top', 'bottom', 'left', 'right'
            compact: options.compact || false,
            theme: options.theme || 'dark',
            showGroups: options.showGroups !== false,
            ...options
        };

        // Indikator-Gruppen definieren
        this.indicatorGroups = [
            {
                name: 'Trend',
                indicators: [
                    { key: 'sma20', label: 'SMA 20', color: '#f59e0b' },
                    { key: 'sma50', label: 'SMA 50', color: '#8b5cf6' },
                    { key: 'sma200', label: 'SMA 200', color: '#ec4899' },
                    { key: 'ema12', label: 'EMA 12', color: '#10b981' },
                    { key: 'ema26', label: 'EMA 26', color: '#ef4444' },
                ]
            },
            {
                name: 'Volatilität',
                indicators: [
                    { key: 'bb', label: 'Bollinger', color: '#64c8ff' },
                ]
            },
            {
                name: 'Momentum',
                indicators: [
                    { key: 'rsi', label: 'RSI', color: '#f59e0b' },
                    { key: 'macd', label: 'MACD', color: '#3b82f6' },
                ]
            },
            {
                name: 'Volume',
                indicators: [
                    { key: 'volume', label: 'Volume', color: '#6b7280' },
                ]
            }
        ];

        // Theme
        this.themes = {
            dark: {
                background: '#1e2130',
                text: '#d1d4dc',
                border: 'rgba(255, 255, 255, 0.1)',
                buttonBg: 'rgba(255, 255, 255, 0.05)',
                buttonHover: 'rgba(255, 255, 255, 0.1)',
                buttonActive: 'rgba(59, 130, 246, 0.3)',
                groupBg: 'rgba(255, 255, 255, 0.03)',
            },
            light: {
                background: '#f8f9fa',
                text: '#131722',
                border: 'rgba(0, 0, 0, 0.1)',
                buttonBg: 'rgba(0, 0, 0, 0.05)',
                buttonHover: 'rgba(0, 0, 0, 0.1)',
                buttonActive: 'rgba(59, 130, 246, 0.2)',
                groupBg: 'rgba(0, 0, 0, 0.03)',
            }
        };

        this.theme = this.themes[this.options.theme] || this.themes.dark;

        // Initialisieren
        this._init();
    }

    /**
     * Initialisiert die Toolbar
     * @private
     */
    _init() {
        this.container.innerHTML = '';

        // Styles einfügen
        this._injectStyles();

        // Toolbar erstellen
        this._createToolbar();

        // Event-Listener hinzufügen
        this._attachEventListeners();
    }

    /**
     * Fügt CSS-Styles ein
     * @private
     */
    _injectStyles() {
        const styleId = 'chart-toolbar-styles';
        if (document.getElementById(styleId)) return;

        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .chart-toolbar {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                padding: 8px 12px;
                background: var(--toolbar-bg);
                border-radius: 6px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 12px;
            }

            .chart-toolbar.compact {
                padding: 4px 8px;
                gap: 4px;
            }

            .chart-toolbar.vertical {
                flex-direction: column;
            }

            .chart-toolbar-group {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 4px 8px;
                background: var(--group-bg);
                border-radius: 4px;
            }

            .chart-toolbar-group-label {
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: var(--text-color);
                opacity: 0.6;
                margin-right: 4px;
                white-space: nowrap;
            }

            .chart-toolbar-btn {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 4px 8px;
                background: var(--button-bg);
                border: 1px solid transparent;
                border-radius: 4px;
                color: var(--text-color);
                cursor: pointer;
                transition: all 0.15s ease;
                font-size: 11px;
                white-space: nowrap;
            }

            .chart-toolbar-btn:hover {
                background: var(--button-hover);
            }

            .chart-toolbar-btn.active {
                background: var(--button-active);
                border-color: var(--indicator-color, #3b82f6);
            }

            .chart-toolbar-btn .indicator-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--indicator-color);
            }

            .chart-toolbar.compact .chart-toolbar-btn {
                padding: 2px 6px;
                font-size: 10px;
            }

            .chart-toolbar.compact .indicator-dot {
                width: 6px;
                height: 6px;
            }

            .chart-toolbar-separator {
                width: 1px;
                height: 24px;
                background: var(--border-color);
                margin: 0 4px;
            }

            .chart-toolbar-presets {
                display: flex;
                gap: 4px;
                margin-left: auto;
            }

            .chart-toolbar-preset-btn {
                padding: 4px 10px;
                background: var(--button-bg);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                color: var(--text-color);
                cursor: pointer;
                font-size: 10px;
                text-transform: uppercase;
                transition: all 0.15s ease;
            }

            .chart-toolbar-preset-btn:hover {
                background: var(--button-hover);
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Erstellt die Toolbar
     * @private
     */
    _createToolbar() {
        this.toolbar = document.createElement('div');
        this.toolbar.className = `chart-toolbar ${this.options.compact ? 'compact' : ''} ${this.options.position === 'left' || this.options.position === 'right' ? 'vertical' : ''}`;
        this.toolbar.style.cssText = `
            --toolbar-bg: ${this.theme.background};
            --text-color: ${this.theme.text};
            --border-color: ${this.theme.border};
            --button-bg: ${this.theme.buttonBg};
            --button-hover: ${this.theme.buttonHover};
            --button-active: ${this.theme.buttonActive};
            --group-bg: ${this.theme.groupBg};
        `;

        // Aktuellen Status vom Chart holen
        const currentStates = this.chart.getIndicatorStates();

        // Gruppen erstellen
        this.indicatorGroups.forEach((group, groupIndex) => {
            if (this.options.showGroups) {
                const groupEl = document.createElement('div');
                groupEl.className = 'chart-toolbar-group';

                const labelEl = document.createElement('span');
                labelEl.className = 'chart-toolbar-group-label';
                labelEl.textContent = group.name;
                groupEl.appendChild(labelEl);

                group.indicators.forEach(indicator => {
                    const btn = this._createIndicatorButton(indicator, currentStates[indicator.key]);
                    groupEl.appendChild(btn);
                });

                this.toolbar.appendChild(groupEl);
            } else {
                group.indicators.forEach(indicator => {
                    const btn = this._createIndicatorButton(indicator, currentStates[indicator.key]);
                    this.toolbar.appendChild(btn);
                });

                // Separator nach jeder Gruppe ausser der letzten
                if (groupIndex < this.indicatorGroups.length - 1 && !this.options.compact) {
                    const separator = document.createElement('div');
                    separator.className = 'chart-toolbar-separator';
                    this.toolbar.appendChild(separator);
                }
            }
        });

        // Presets hinzufügen
        if (!this.options.compact) {
            const presetsContainer = document.createElement('div');
            presetsContainer.className = 'chart-toolbar-presets';

            const presets = [
                { key: 'none', label: 'Keine' },
                { key: 'basic', label: 'Basic' },
                { key: 'full', label: 'Alle' },
            ];

            presets.forEach(preset => {
                const btn = document.createElement('button');
                btn.className = 'chart-toolbar-preset-btn';
                btn.textContent = preset.label;
                btn.dataset.preset = preset.key;
                presetsContainer.appendChild(btn);
            });

            this.toolbar.appendChild(presetsContainer);
        }

        this.container.appendChild(this.toolbar);
    }

    /**
     * Erstellt einen Indikator-Button
     * @private
     */
    _createIndicatorButton(indicator, isActive) {
        const btn = document.createElement('button');
        btn.className = `chart-toolbar-btn ${isActive ? 'active' : ''}`;
        btn.style.setProperty('--indicator-color', indicator.color);
        btn.dataset.indicator = indicator.key;

        const dot = document.createElement('span');
        dot.className = 'indicator-dot';
        dot.style.background = indicator.color;
        btn.appendChild(dot);

        const label = document.createElement('span');
        label.textContent = indicator.label;
        btn.appendChild(label);

        return btn;
    }

    /**
     * Fügt Event-Listener hinzu
     * @private
     */
    _attachEventListeners() {
        // Indikator-Buttons
        this.toolbar.querySelectorAll('.chart-toolbar-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const indicator = btn.dataset.indicator;
                const isActive = btn.classList.toggle('active');
                this.chart.toggleIndicator(indicator, isActive);

                // Custom Event auslösen
                this.container.dispatchEvent(new CustomEvent('indicatorToggle', {
                    detail: { indicator, enabled: isActive }
                }));
            });
        });

        // Preset-Buttons
        this.toolbar.querySelectorAll('.chart-toolbar-preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const preset = btn.dataset.preset;
                this._applyPreset(preset);
            });
        });
    }

    /**
     * Wendet ein Preset an
     * @private
     */
    _applyPreset(preset) {
        const presets = {
            none: {
                sma20: false, sma50: false, sma200: false,
                ema12: false, ema26: false, bb: false,
                rsi: false, macd: false, volume: false
            },
            basic: {
                sma20: true, sma50: true, sma200: false,
                ema12: false, ema26: false, bb: false,
                rsi: true, macd: false, volume: true
            },
            full: {
                sma20: true, sma50: true, sma200: true,
                ema12: false, ema26: false, bb: true,
                rsi: true, macd: true, volume: true
            }
        };

        const settings = presets[preset];
        if (!settings) return;

        Object.entries(settings).forEach(([indicator, enabled]) => {
            this.chart.toggleIndicator(indicator, enabled);

            // Button-Status aktualisieren
            const btn = this.toolbar.querySelector(`[data-indicator="${indicator}"]`);
            if (btn) {
                btn.classList.toggle('active', enabled);
            }
        });

        // Custom Event auslösen
        this.container.dispatchEvent(new CustomEvent('presetApplied', {
            detail: { preset, settings }
        }));
    }

    /**
     * Aktualisiert den Status eines Indikators
     * @param {string} indicator
     * @param {boolean} enabled
     */
    setIndicatorState(indicator, enabled) {
        const btn = this.toolbar.querySelector(`[data-indicator="${indicator}"]`);
        if (btn) {
            btn.classList.toggle('active', enabled);
        }
    }

    /**
     * Gibt alle aktiven Indikatoren zurück
     * @returns {string[]}
     */
    getActiveIndicators() {
        return Array.from(this.toolbar.querySelectorAll('.chart-toolbar-btn.active'))
            .map(btn => btn.dataset.indicator);
    }

    /**
     * Zerstört die Toolbar
     */
    destroy() {
        this.container.innerHTML = '';
    }
}

// Export für Module
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChartToolbar;
}
