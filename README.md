# E201 GUI

---

## Overview
Simple GUI to check encoder reading. Supports plotting error and noise.


### Supported Features
- Supported E201:
  - E201-B
  - E201-Q


- Motor types:
  - EPOS

---

# Prerequisites

Before installation, ensure **E201 GUI repository** is cloned:
- Repository: [E210 GUI Repository](https://rls-git/Laboratorij/E201_gui.git)
- Cloning Help: [FWSW wiki](https://rls-git/FWSW/project_generator/wiki/Development-environment-setup#git-repo)

---

# Interface running 
### USING PYTHON 3.11 FROM SOFTWARE CENTER
1. Install uv to python: `pip install uv`

2. Run prepared bat file in cloned repo: ```_run_Gui_python311.bat```

### USING FWSW APPS
  1.  Install **FWSW_APPS**
    - Installation Guide: [FWSW wiki](https://rls-git/FWSW/project_generator/wiki/Development-environment-setup#fwsw_apps)

  2. Run prepared bat file in cloned repo: ```__run_Gui_fwsw_apps.bat```
    _run_gui.bat

---

# Application Usage

### Connection

- Choose correct comport (if not found on UI initialization, click *REFRESH PORTS*)
- Choose E201 type
- In *DUT/REF SETUP* setup encoder configuration (for now you need to do it before encoder initialization!)
After launching the application, configure:


### Plot control
- Synchronized reading - if triggers of E201's are connected, initialize trigger reading
- Invert DUT direction - choose if DUT and reference are counting in differend directions
- Positions plot - supports display of DUT, REF or DUT&REF. Units setup (deg/µm) also supported
- Analyse diagram - supports display of error and noise (DUT or REF)
- By click on *ZERO OFFSET* you can reset error to zero