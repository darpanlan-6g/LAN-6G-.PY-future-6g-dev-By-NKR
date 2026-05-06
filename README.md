# 🛰️ 6G Network Digital Twin with Deep Reinforcement Learning

> **Research Engineer:** Nitish Kumar Roy  
> **Domain:** 6G Telecommunications · Network Digital Twins · Deep Reinforcement Learning  
> **Status:** 🔬 Active Research | Framework Design & Simulation Phase

---

## 📌 Overview

This repository contains the architectural design, simulation code, and implementation framework for a **7-Layer 6G Network Digital Twin (NDT)** integrated with **Deep Reinforcement Learning (DRL)**. The system creates a high-fidelity virtual replica of a physical 6G network, enabling autonomous, sub-millisecond resource orchestration without risk to live users.

6G networks operate at **Terahertz (THz) frequencies** with massive MIMO arrays and high-velocity mobility patterns — environments where traditional static optimization becomes computationally irreducible. This project addresses that challenge through a closed-loop digital twin architecture where a PPO-based RL agent is trained in a risk-free virtual testbed before deployment to physical infrastructure.

---

## 🏗️ Architecture: 7-Layer Framework

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 6 │  Evaluation & Visualization (MSE/MAE, CDF, TensorBoard)  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5 │  Control & Actuation (Beamforming, TX Power, Slicing)     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4 │  Joint Training Pipeline (4-Phase: Offline → Deploy)      │
├──────────────────────────────┬──────────────────────────────────┤
│  Layer 3 │  Model Registry & Experience Replay Buffer               │
├──────────────────────────────┴──────────────────────────────────┤
│  Layer 2B│  Dynamic Twin — PPO Agent (Actor-Critic + GAE)           │
│  Layer 2A│  Static Twin — Physics Models + LSTM Channel Predictor   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 │  Data Ingestion & Feature Engineering (ZeroMQ/Protobuf)  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 0 │  Physical Network — 6G gNodeBs, UEs, ns-3 Simulator      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Layer-by-Layer Description

### Layer 0 — Physical Network & Real-Time Telemetry
- **Source:** 6G gNodeB base stations / ns-3 discrete-event simulator
- **Telemetry:** SINR, throughput, handover logs, (x, y) position, ICS data
- **Tools:** OpenAirInterface (OAI), SDR platforms

### Layer 1 — Data Ingestion & Feature Engineering
- **Stream:** ZeroMQ socket streams or historical CSV logs
- **Processing:** Normalization, sliding-window buffer, state vector construction
- **Serialization:** Protocol Buffers (Protobuf) for C++ ↔ Python interop

### Layer 2A — Static Digital Twin (Physics-Aware)
| Component | Algorithm | Role |
|---|---|---|
| Physics Model | Friis + ITU-R P.676 | THz/mmWave path loss & attenuation |
| Channel Predictor | LSTM / GRU | Future channel state prediction |
| Coverage Map | SINR Heatmap Grid | Spatial signal quality visualization |
| Anomaly Detector | Isolation Forest | Base station failure & traffic surge detection |

### Layer 2B — Dynamic Digital Twin (RL Core)
- **Framework:** OpenAI Gym MDP formulation
- **Agent:** Proximal Policy Optimization (PPO) with Actor-Critic + GAE

**State Space:** `[SINR, Throughput, Position(x,y), Handover_Status]`

**Action Space:**
- Beamforming gain: 0–44 dBi
- TX power: 23–46 dBm
- Handover triggers
- Network slicing commands (eMBB / URLLC)

**Reward Function:**
```
R = 0.4·norm_SINR + 0.3·norm_TP − 0.2·norm_latency − 0.1·HO_penalty
```

### Layer 3 — Model Registry & Experience Replay
- Stores LSTM weights and PPO policy weights
- Experience Replay Buffer: `(s, a, r, s')` transitions
- Checkpointing and TensorBoard logging

### Layer 4 — Joint Training Pipeline (4 Phases)
| Phase | Description |
|---|---|
| Phase 1 | Offline supervised learning on historical datasets |
| Phase 2 | RL in Static Twin (risk-free exploration) |
| Phase 3 | RL in Live Twin (sandboxed real-time telemetry) |
| Phase 4 | Deployment to physical network |

### Layer 5 — Control & Actuation
- Beamforming steering for massive MIMO
- TX power control (inter-cell interference minimization)
- Dynamic network slicing (eMBB vs. URLLC partitioning)

### Layer 6 — Evaluation & Visualization
- Twin Fidelity: MSE / MAE between virtual and physical SINR
- SINR CDF curves, RL convergence graphs, live Matplotlib plots
- Auto-triggers model retraining when MSE exceeds threshold

---

## ⚙️ Synchronization: Age of Digital Twin (AoDT)

```
AoDT = t − t_update
```

Maintaining a **low AoDT** is critical — stale state information leads to outdated beam steering and dropped connections. The framework uses **TD3 (Twin Delayed DDPG)** to optimize telemetry update frequency under resource constraints.

---

## 🛠️ Implementation Toolchains

### Option A: ns-3 + ns3-gym (Standard Academic)
```
Simulator  : ns-3 (5G-LENA / mmWave modules)
RL Bridge  : ns3-gym (OpenAI Gym wrapper)
IPC        : ZeroMQ + Protobuf
Analytics  : TensorBoard + Matplotlib
```
Best for: Layer 2B RL training, MAC/PHY protocol interaction

### Option B: NVIDIA Aerial + Sionna (High-Fidelity Physics)
```
Ray Tracing : NVIDIA Sionna RT (3D deterministic channel modeling)
Platform    : NVIDIA Aerial Omniverse Digital Twin (AODT)
Acceleration: CUDA / Tensor Cores (LDPC decoding, neural receivers)
```
Best for: Layer 2A static twin — SINR heatmaps, city-scale RF environments

---

## 🚀 Quick Start

### Prerequisites
```bash
# System (Ubuntu 22.04 recommended)
sudo apt-get install python3 pip3 g++
sudo apt-get install libzmq5 libzmq5-dev
sudo apt-get install libprotobuf-dev protobuf-compiler
```

### ns-3 + ns3-gym Setup
```bash
# Clone ns-3 (v3.36+)
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev

# Add ns3-gym to contrib
cp -r ns3-gym/src contrib/opengym

# Build
./ns3 configure --enable-examples
./ns3 build

# Install Python module
cd contrib/opengym/model/ns3gym
pip3 install .
```

### Training the PPO Agent
```python
from stable_baselines3 import PPO
import gym

# Initialize custom ns-3 gym environment
env = gym.make("6GNetworkTwin-v0")

# Define PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    gae_lambda=0.95,
    clip_range=0.2,
    n_steps=2048,
    batch_size=64
)

# Phase 2: Train in static twin
model.learn(total_timesteps=1_000_000)
model.save("ppo_6g_static_twin")
```

### Running the Simulation
```bash
# Terminal 1: Start ns-3 simulator (opens ZMQ port 5555)
./ns3 run "6g-digital-twin-scenario"

# Terminal 2: Start RL agent
python3 train_ppo_agent.py
```

---

## 📐 Reward Function Implementation
```python
def compute_reward(sinr, throughput, latency, handover_count):
    norm_sinr     = (sinr - SINR_MIN) / (SINR_MAX - SINR_MIN)
    norm_tp       = (throughput - TP_MIN) / (TP_MAX - TP_MIN)
    norm_latency  = (latency - LAT_MIN) / (LAT_MAX - LAT_MIN)
    ho_penalty    = min(handover_count / MAX_HO, 1.0)

    reward = (0.4 * norm_sinr
            + 0.3 * norm_tp
            - 0.2 * norm_latency
            - 0.1 * ho_penalty)
    return reward
```

---

## 🔭 Future Work

- [ ] **Federated Digital Twins** — Distributed twin synchronization across multi-operator 6G deployments
- [ ] **Goal-Oriented Semantic Twinning (GOST)** — Lightweight task-specific modeling for SAGSIN (Space-Air-Ground-Sea Integrated Networks)
- [ ] **Multi-Agent RL** — Coordinated beamforming across multiple base stations with MARL
- [ ] **THz Channel Modeling** — Integrating molecular absorption effects beyond ITU-R P.676
- [ ] **Sim-to-Real Transfer** — Domain randomization techniques to close the sim-to-real gap
- [ ] **AoDT-Aware Scheduling** — Dynamic telemetry update policies using TD3/SAC
- [ ] **Security Layer** — Adversarial attack detection (e.g., Sandwich attacks) integrated into the anomaly subsystem
- [ ] **6G NTN Support** — Non-Terrestrial Network (satellite/UAV) node integration in ns-3
- [ ] **NVIDIA Sionna RT Integration** — City-scale deterministic ray tracing linked to the RL training loop
- [ ] **Digital Twin Marketplace** — Pre-trained twin models for standard 3GPP 6G deployment scenarios

---

## 📊 Key Performance Targets

| KPI | 6G Target | Architecture Role |
|---|---|---|
| Latency | < 1 ms | Reward penalty term |
| SINR Optimization | Maximize | Primary reward (weight: 0.4) |
| Throughput | Maximize | Secondary reward (weight: 0.3) |
| Handover Rate | Minimize | Penalty term (weight: 0.1) |
| Twin Fidelity (MSE) | < threshold | Layer 6 auto-retraining trigger |
| AoDT | → 0 | TD3-optimized telemetry scheduling |

---

## 📚 Key References

1. IEEE Xplore — Network Digital Twin for 6G and Beyond (End-to-End Framework)
2. ns3-gym — OpenAI Gym integration for ns-3
3. NVIDIA Aerial Omniverse Digital Twin (AODT)
4. ITU-R P.676 — Attenuation by atmospheric gases (THz-critical)
5. Proximal Policy Optimization (Schulman et al., 2017)
6. Age of Digital Twin synchronization under budget constraints

---

## 👨‍💻 Author

**Nitish Kumar Roy**  
Research Engineer — 6G Networks & Intelligent Systems  
*Specialization: Network Digital Twins · Deep Reinforcement Learning · THz Communications*

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> *"The integration of Digital Twins and Reinforcement Learning in 6G is not merely a simulation enhancement — it is a fundamental requirement for operating at the edge of physical possibility."*
