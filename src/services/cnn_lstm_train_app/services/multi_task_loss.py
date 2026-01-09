"""
Multi-Task Loss Function für CNN-LSTM Training.

Kombinierte Loss-Funktion fuer:
- Preis-Vorhersage (MSE + Direction Penalty)
- Pattern-Klassifikation (BCE Multi-Label)
- Regime-Vorhersage (CrossEntropy)
"""

from dataclasses import dataclass
from typing import Optional

# Lazy import fuer PyTorch
torch = None
nn = None
F = None


def _ensure_torch():
    """Stellt sicher dass PyTorch geladen ist."""
    global torch, nn, F
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch = _torch
        nn = _nn
        F = _F


@dataclass
class LossWeights:
    """Gewichtungen fuer Multi-Task Loss."""
    price: float = 0.4
    pattern: float = 0.35
    regime: float = 0.25

    def __post_init__(self):
        """Validiert dass Gewichtungen summieren zu 1."""
        total = self.price + self.pattern + self.regime
        if abs(total - 1.0) > 0.01:
            # Normalisiere
            self.price /= total
            self.pattern /= total
            self.regime /= total


@dataclass
class LossComponents:
    """Container fuer Loss-Komponenten."""
    total: float
    price: float
    pattern: float
    regime: float
    direction_accuracy: Optional[float] = None


def create_multi_task_loss(weights: Optional[LossWeights] = None):
    """
    Factory-Funktion fuer Multi-Task Loss.

    Args:
        weights: Gewichtungen fuer die einzelnen Tasks

    Returns:
        MultiTaskLoss Modul
    """
    _ensure_torch()

    if weights is None:
        weights = LossWeights()

    class MultiTaskLoss(nn.Module):
        """
        Multi-Task Loss mit gewichteter Kombination.

        Loss = w_price * L_price + w_pattern * L_pattern + w_regime * L_regime

        Wobei:
        - L_price: MSE + Direction Penalty
        - L_pattern: BCE with Logits (Multi-Label)
        - L_regime: CrossEntropy (Multi-Class)
        """

        def __init__(self):
            super().__init__()
            self.weights = weights

            # Loss Functions
            self.mse_loss = nn.MSELoss()
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.ce_loss = nn.CrossEntropyLoss()

            # Direction Penalty Weight
            self.direction_penalty_weight = 0.3

        def price_loss(
            self,
            predictions: "torch.Tensor",
            targets: "torch.Tensor"
        ) -> tuple["torch.Tensor", float]:
            """
            Berechnet Preis-Loss mit Direction Penalty.

            Args:
                predictions: Vorhersagen (batch, num_horizons)
                targets: Targets (batch, num_horizons)

            Returns:
                (loss, direction_accuracy)
            """
            # MSE Loss
            mse = self.mse_loss(predictions, targets)

            # Direction Accuracy (Sign Agreement)
            pred_direction = torch.sign(predictions)
            target_direction = torch.sign(targets)
            direction_correct = (pred_direction == target_direction).float()
            direction_accuracy = direction_correct.mean().item()

            # Direction Penalty: Bestrafe falsche Richtungen
            direction_penalty = (1.0 - direction_correct).mean()

            # Kombiniere
            loss = mse + self.direction_penalty_weight * direction_penalty

            return loss, direction_accuracy

        def pattern_loss(
            self,
            predictions: "torch.Tensor",
            targets: "torch.Tensor"
        ) -> "torch.Tensor":
            """
            Berechnet Pattern-Loss (Multi-Label BCE).

            Args:
                predictions: Logits (batch, num_patterns)
                targets: Binary Labels (batch, num_patterns)

            Returns:
                BCE Loss
            """
            return self.bce_loss(predictions, targets)

        def regime_loss(
            self,
            predictions: "torch.Tensor",
            targets: "torch.Tensor"
        ) -> "torch.Tensor":
            """
            Berechnet Regime-Loss (CrossEntropy).

            Args:
                predictions: Logits (batch, num_classes)
                targets: Class Indices (batch,)

            Returns:
                CrossEntropy Loss
            """
            return self.ce_loss(predictions, targets)

        def forward(
            self,
            predictions: dict,
            targets: dict
        ) -> tuple["torch.Tensor", LossComponents]:
            """
            Forward pass - berechnet kombinierten Loss.

            Args:
                predictions: Dict mit 'price', 'patterns', 'regime' Tensoren
                targets: Dict mit entsprechenden Target-Tensoren

            Returns:
                (total_loss, LossComponents)
            """
            # Price Loss
            l_price, dir_acc = self.price_loss(
                predictions['price'],
                targets['price']
            )

            # Pattern Loss
            l_pattern = self.pattern_loss(
                predictions['patterns'],
                targets['patterns']
            )

            # Regime Loss
            l_regime = self.regime_loss(
                predictions['regime'],
                targets['regime']
            )

            # Kombiniere mit Gewichtungen
            total_loss = (
                self.weights.price * l_price +
                self.weights.pattern * l_pattern +
                self.weights.regime * l_regime
            )

            components = LossComponents(
                total=total_loss.item(),
                price=l_price.item(),
                pattern=l_pattern.item(),
                regime=l_regime.item(),
                direction_accuracy=dir_acc
            )

            return total_loss, components

        def get_weights(self) -> LossWeights:
            """Gibt aktuelle Gewichtungen zurueck."""
            return self.weights

        def set_weights(self, new_weights: LossWeights):
            """Setzt neue Gewichtungen."""
            self.weights = new_weights

    return MultiTaskLoss()


class FocalLoss:
    """
    Focal Loss fuer unbalancierte Pattern-Klassifikation.

    FL(p) = -alpha * (1-p)^gamma * log(p)

    Reduziert den Einfluss von leicht klassifizierbaren Samples.
    """

    @staticmethod
    def create(alpha: float = 0.25, gamma: float = 2.0):
        """Erstellt Focal Loss Modul."""
        _ensure_torch()

        class _FocalLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
                """
                Args:
                    inputs: Logits (batch, num_classes)
                    targets: Binary Labels (batch, num_classes)
                """
                bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                pt = torch.exp(-bce)  # p_t
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
                return focal_loss.mean()

        return _FocalLoss()


class LabelSmoothingCrossEntropy:
    """
    Cross Entropy mit Label Smoothing fuer Regime-Klassifikation.

    Verhindert Overconfidence und verbessert Generalisierung.
    """

    @staticmethod
    def create(smoothing: float = 0.1):
        """Erstellt Label Smoothing CE Modul."""
        _ensure_torch()

        class _LabelSmoothingCE(nn.Module):
            def __init__(self):
                super().__init__()
                self.smoothing = smoothing

            def forward(self, inputs: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
                """
                Args:
                    inputs: Logits (batch, num_classes)
                    targets: Class Indices (batch,)
                """
                num_classes = inputs.size(-1)
                log_probs = F.log_softmax(inputs, dim=-1)

                # Smooth Labels
                with torch.no_grad():
                    smooth_targets = torch.zeros_like(log_probs)
                    smooth_targets.fill_(self.smoothing / (num_classes - 1))
                    smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

                loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
                return loss

        return _LabelSmoothingCE()


def create_advanced_multi_task_loss(
    weights: Optional[LossWeights] = None,
    use_focal_loss: bool = True,
    use_label_smoothing: bool = True,
    focal_gamma: float = 2.0,
    smoothing: float = 0.1
):
    """
    Factory fuer erweiterte Multi-Task Loss mit Focal Loss und Label Smoothing.

    Args:
        weights: Task-Gewichtungen
        use_focal_loss: Focal Loss fuer Patterns verwenden
        use_label_smoothing: Label Smoothing fuer Regime verwenden
        focal_gamma: Gamma fuer Focal Loss
        smoothing: Smoothing Factor

    Returns:
        Advanced MultiTaskLoss Modul
    """
    _ensure_torch()

    if weights is None:
        weights = LossWeights()

    class AdvancedMultiTaskLoss(nn.Module):
        """Erweiterte Multi-Task Loss mit optionalen Verbesserungen."""

        def __init__(self):
            super().__init__()
            self.weights = weights

            # Price Loss
            self.mse_loss = nn.MSELoss()
            self.direction_penalty_weight = 0.3

            # Pattern Loss
            if use_focal_loss:
                self.pattern_loss_fn = FocalLoss.create(gamma=focal_gamma)
            else:
                self.pattern_loss_fn = nn.BCEWithLogitsLoss()

            # Regime Loss
            if use_label_smoothing:
                self.regime_loss_fn = LabelSmoothingCrossEntropy.create(smoothing)
            else:
                self.regime_loss_fn = nn.CrossEntropyLoss()

        def forward(self, predictions: dict, targets: dict) -> tuple["torch.Tensor", LossComponents]:
            # Price Loss mit Direction
            mse = self.mse_loss(predictions['price'], targets['price'])
            pred_dir = torch.sign(predictions['price'])
            targ_dir = torch.sign(targets['price'])
            dir_correct = (pred_dir == targ_dir).float()
            dir_acc = dir_correct.mean().item()
            dir_penalty = (1.0 - dir_correct).mean()
            l_price = mse + self.direction_penalty_weight * dir_penalty

            # Pattern Loss
            l_pattern = self.pattern_loss_fn(predictions['patterns'], targets['patterns'])

            # Regime Loss
            l_regime = self.regime_loss_fn(predictions['regime'], targets['regime'])

            # Kombiniere
            total = (
                self.weights.price * l_price +
                self.weights.pattern * l_pattern +
                self.weights.regime * l_regime
            )

            return total, LossComponents(
                total=total.item(),
                price=l_price.item(),
                pattern=l_pattern.item(),
                regime=l_regime.item(),
                direction_accuracy=dir_acc
            )

    return AdvancedMultiTaskLoss()
