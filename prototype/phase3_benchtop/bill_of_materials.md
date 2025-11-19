# PhotonCore-X Bench-Top Prototype BOM

## Bill of Materials for Tabletop Optical Computing Demo

### Total Estimated Cost: $15,000 - $25,000

---

## 1. Light Source ($3,000 - $5,000)

| Item | Qty | Vendor | Part Number | Price |
|------|-----|--------|-------------|-------|
| Tunable C-Band Laser | 1 | Thorlabs | TLX1 | $3,500 |
| Fiber Patch Cables (PM) | 10 | Thorlabs | P3-1550PM-FC-2 | $80 ea |
| Fiber Splitter 1x8 | 1 | Thorlabs | TN1550R5A1 | $450 |

**Alternative (Lower Cost):**
| Item | Qty | Vendor | Part Number | Price |
|------|-----|--------|-------------|-------|
| DFB Laser 1550nm | 1 | Thorlabs | DJ532-10 | $800 |
| Temperature Controller | 1 | Thorlabs | TED200C | $600 |

---

## 2. Modulators ($4,000 - $8,000)

| Item | Qty | Vendor | Part Number | Price |
|------|-----|--------|-------------|-------|
| LiNbO3 MZM 10GHz | 4 | iXBlue | MXAN-LN-10 | $1,200 ea |
| Phase Modulator | 4 | iXBlue | MPZ-LN-10 | $800 ea |
| RF Amplifiers | 4 | Mini-Circuits | ZHL-1-2W+ | $150 ea |

**For 8-port demo (28 MZIs):**
- Use fiber-based variable attenuators for amplitude
- Phase modulators for phase control
- Total: ~$6,000

---

## 3. Detection ($2,000 - $4,000)

| Item | Qty | Vendor | Part Number | Price |
|------|-----|--------|-------------|-------|
| InGaAs Photodetector | 8 | Thorlabs | DET08CFC | $250 ea |
| Balanced Detector | 4 | Thorlabs | PDB450C | $800 ea |
| Transimpedance Amp | 8 | Analog Devices | AD8015 | $20 ea |

---

## 4. Control Electronics ($2,000 - $4,000)

| Item | Qty | Vendor | Part Number | Price |
|------|-----|--------|-------------|-------|
| FPGA Dev Board | 1 | Digilent | Arty A7-100T | $250 |
| DAC Board (16-ch) | 2 | Analog Devices | AD5370 Eval | $400 ea |
| ADC Board (8-ch) | 1 | Texas Instruments | ADS8688 EVM | $300 |
| Power Supply | 1 | Keysight | E36312A | $1,500 |

---

## 5. Optical Components ($2,000 - $3,000)

| Item | Qty | Vendor | Part Number | Price |
|------|-----|--------|-------------|-------|
| Fiber Couplers 50:50 | 20 | Thorlabs | TW1550R5F1 | $80 ea |
| Fiber Delays (1m) | 10 | Thorlabs | P1-SMF28E-FC-1 | $50 ea |
| Polarization Controller | 4 | Thorlabs | FPC560 | $250 ea |
| Isolator | 2 | Thorlabs | IO-H-1550APC | $400 ea |

---

## 6. Optomechanics ($1,000 - $2,000)

| Item | Qty | Vendor | Part Number | Price |
|------|-----|--------|-------------|-------|
| Optical Breadboard | 1 | Thorlabs | MB3045/M | $500 |
| Fiber Mounts | 20 | Thorlabs | HFB001 | $30 ea |
| Posts and Clamps | Set | Thorlabs | - | $300 |

---

## Assembly Diagram

```
                    ┌─────────────┐
                    │   Laser     │
                    │  1550 nm    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  1x8 Split  │
                    └──────┬──────┘
                           │
         ┌─────┬─────┬─────┼─────┬─────┬─────┬─────┐
         │     │     │     │     │     │     │     │
       ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐
       │MOD│ │MOD│ │MOD│ │MOD│ │MOD│ │MOD│ │MOD│ │MOD│
       └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
         │     │     │     │     │     │     │     │
       ┌─┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─┐
       │                                             │
       │            MZI MESH (28 couplers)           │
       │                                             │
       └─┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─┘
         │     │     │     │     │     │     │     │
       ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐
       │DET│ │DET│ │DET│ │DET│ │DET│ │DET│ │DET│ │DET│
       └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
         │     │     │     │     │     │     │     │
       ┌─┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─┐
       │              FPGA Control                   │
       └─────────────────┬───────────────────────────┘
                         │
                    ┌────▼────┐
                    │   PC    │
                    └─────────┘
```

---

## Vendors and Contacts

### Primary Vendors:
- **Thorlabs**: www.thorlabs.com (Most optical components)
- **iXBlue**: www.ixblue.com (High-speed modulators)
- **Digilent**: www.digilent.com (FPGA boards)
- **Mini-Circuits**: www.minicircuits.com (RF components)

### Alternative Sources:
- **Edmund Optics**: Optical breadboards, mounts
- **Newport**: Precision optomechanics
- **Gooch & Housego**: Acousto-optic modulators

---

## Assembly Notes

### Phase 1: Basic 2x2 Demo
1. One laser, one splitter, one coupler, two detectors
2. Verify interference and phase control
3. Cost: ~$3,000

### Phase 2: 4-port Mesh
1. Add modulators and more couplers
2. Test Clements decomposition
3. Cost: ~$8,000

### Phase 3: Full 8-port Demo
1. Complete mesh with all 28 MZIs
2. FPGA control integration
3. Cost: ~$20,000

---

## Test Plan

1. **Insertion Loss**: Measure total optical loss through mesh
2. **Extinction Ratio**: Verify MZI switching contrast >20dB
3. **Phase Accuracy**: Compare programmed vs measured phases
4. **Matrix Fidelity**: Implement known unitary, measure output
5. **Speed**: Time for full matrix reconfiguration
6. **Stability**: Monitor drift over time

---

## Safety Considerations

- Class 3B laser (1550nm, >5mW)
- Required: Laser safety training, goggles
- Fiber connectors: Never look into fiber end
- Electrical: Proper grounding for RF equipment
