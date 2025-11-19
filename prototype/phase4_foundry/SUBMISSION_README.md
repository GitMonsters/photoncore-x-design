
# PhotonCore-X Foundry Submission

## Target Process
- GlobalFoundries 45SPCLO (45nm Silicon Photonics)
- Alternative: IMEC iSiPP50G

## Files
- photoncore_8port.gds: 8-port test chip (28 MZIs)
- photoncore_16port.gds: 16-port demo chip (120 MZIs)
- photoncore_64port.gds: Full 64-port chip (2016 MZIs)

## Design Rules
- Waveguide width: 500nm
- Minimum gap: 200nm
- Minimum bend radius: 5μm
- Heater length: 100μm (π phase shift)

## Layers
- Layer 1: Si ridge waveguide (220nm)
- Layer 10: TiN heater
- Layer 20: Metal 1
- Layer 40: Grating etch

## MPW Options
1. AIM Photonics (Albany, NY)
   - Cost: ~$30K for 5x5mm
   - Turnaround: 3 months

2. IMEC (Leuven, Belgium)
   - Cost: ~$20K for 2x2mm
   - Turnaround: 4 months

3. CompoundTek (Singapore)
   - Cost: ~$15K for 3x3mm
   - Turnaround: 2 months

## Testing Requirements
- Fiber array coupling (127μm pitch)
- DC probes for heater control
- Expected insertion loss: <3dB per MZI
- Expected phase efficiency: 25 mW/π

## Contact
- Email: tapeout@photoncore.ai
- Phone: +1-xxx-xxx-xxxx
