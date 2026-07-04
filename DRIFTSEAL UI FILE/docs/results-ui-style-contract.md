# Results UI Style Contract (Dashboard)

This document defines the visual contract for the **Results / Dashboard** view (the page shown after uploading `dataset` + `metrics` (+ optional `model.py`) and completing a DriftSeal scan).

## Scope

Applies to:

- `Dashboard` route (`/dashboard`) and the **two-card Results layout**.
- The **Behavioral Drift Matrix** (Accuracy/Precision/Recall/F1-Score/AUC-ROC across Clean/Flip/Backdoor/Evasion).
- The **Poison Detection** card (gauge, poison meter, threshold, detector sub-scores, drift score).

Non-goals:

- Changing feature logic or data sources.
- Replacing components or introducing a new theme system.

## Non-Negotiable UI Invariants

### Layout

- Left sidebar with icon + label items in a vertical stack.
- Main content uses a two-column grid on desktop:
  - Left: **Poison Detection** card.
  - Right: **Behavioral Drift Matrix** card.

### Visual Language

- Pastel palette, soft shadows, and rounded corners.
- Results are shown using:
  - `RaisedCard` / `InsetPanel` styling.
  - Pill/badge patterns (rounded, soft shadow) for status and metric cells.

### Behavioral Drift Matrix

- Must keep these rows and column semantics:
  - Rows: `Accuracy`, `Precision`, `Recall`, `F1-Score`, `AUC-ROC`
  - Columns: `Clean`, `Flip`, `Backdoor`, `Evasion`
- Each cell is displayed as a rounded chip with a value.
- Color mapping must remain consistent:
  - Excellent `> 0.90` (green)
  - Good `0.85–0.90` (blue)
  - Warning `0.80–0.85` (orange)
  - Critical `< 0.80` (red)
- Legend must reflect the same thresholds and colors.

### Poison Detection Card

- Gauge remains circular with a soft track + colored progress arc.
- Must include:
  - Poison meter bar
  - Model threshold
  - Detector sub-scores: `Autoencoder`, `Isolation Forest`, `One-Class SVM`
  - Drift score

## Implementation Notes (to preserve style)

- Do not rename/remove `RaisedCard`, `InsetPanel`, `RiskBadge`, `GaugeWidget` usage in `DashboardPage`.
- Only change values/data-binding logic; do not change structure, spacing classes, or color tokens.

## Primary Files

- `src/app/pages/DashboardPage.tsx`
- `src/app/components/skeuomorphic/RaisedCard.tsx`
- `src/app/components/skeuomorphic/InsetPanel.tsx`
- `src/app/components/skeuomorphic/RiskBadge.tsx`
- `src/app/components/skeuomorphic/GaugeWidget.tsx`

