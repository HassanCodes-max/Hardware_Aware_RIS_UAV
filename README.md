# Hardware-Aware RIS UAV

A small Python baseline for evaluating a UAV-to-user link assisted by a reconfigurable intelligent surface (RIS). The current model includes a circular UAV trajectory, free-space path loss, RIS element gain, reflection efficiency, and spectral efficiency.

## Project structure

- `main.py`: runs the simulation and writes `results/simulation.csv`
- `config.py`: shared simulation parameters
- `trajectory.py`: UAV trajectory generation
- `channel.py`: hardware-aware RIS channel calculations
- `results/`: generated simulation output

## Run

```powershell
cd Hardware_Aware_RIS_UAV
python -m pip install -r requirements.txt
python main.py
```

The generated CSV contains the UAV position, received power, and spectral efficiency for each time step.
