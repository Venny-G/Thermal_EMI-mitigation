# Clustered Electric Propulsion Risk Mapper

This repository contains a simplified, physics-informed simulator for showing how clustered electric propulsion layouts create:

- thermal loading zones,
- EMI / magnetic interference zones,
- subsystem exposure risks,
- and layout or shielding recommendations.

The current MVP focuses on a 2D spacecraft bus view using transparent heuristic models:

- thermal load is represented as directional plume-shaped decay plus a local near-field hotspot,
- EMI is represented as a near-thruster magnetic term plus a directional tail,
- subsystem status is classified from threshold ratios into `safe`, `caution`, or `critical`.

These fields are engineering proxies, not high-fidelity CFD or Maxwell solutions. They are meant to support fast layout trade studies and design storytelling.

## Quick start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the dashboard:

```bash
streamlit run app.py
```

The app includes three presets:

- `Single Thruster Baseline`
- `Dual Thruster Interaction`
- `2x2 Clustered Configuration`

## Current scope

The MVP demonstrates:

- spacecraft bus and subsystem placement,
- 1 to 4 thruster presets loaded from YAML,
- thermal, EMI, and combined risk maps,
- subsystem exposure tables,
- simple failure warnings,
- actionable recommendations for spacing, shielding, and placement.

## Modeling assumptions

- The simulator works in a 2D spacecraft layout plane.
- Thruster influence is directional and decays with distance.
- Overlapping fields accumulate linearly.
- Shielding reduces the effective exposure seen by a subsystem.
- All outputs are proxy indices intended for relative comparison and early integration studies.

## Testing

Run the lightweight validation suite with:

```bash
pytest
```
