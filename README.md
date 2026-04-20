# DRL Dynamic Positioning with Explainable AI

Deep Reinforcement Learning (PPO) for Dynamic Positioning of the milliAmpere1 autonomous ferry at NTNU, with a real-time SHAP-based Explainable AI dashboard.

- **Author:** Alexander Sandberg
- **Course:** TTK4900, Cybernetics and Robotics @ NTNU
- **Paper:** [Deep Reinforcement Learning for Ship Dynamic Positioning: A Live Explainable AI Approach on an Autonomous Ferry Prototype](https://ieeexplore.ieee.org/document/11480113)
- **Thesis:** [Deploying Trustworthy Deep Reinforcement Learning for Dynamic Positioning: A Real-time Explainable AI Approach on Real Maritime Cyber-Physical Systems](https://nva.sikt.no/registration/019a1506973c-03176847-b705-4208-9048-5069284d9c61)

If you use this work, please cite:
```bibtex
@article{sandberg2026drl,
  title   = {Deep Reinforcement Learning for Ship Dynamic Positioning: A Live Explainable AI Approach on an Autonomous Ferry Prototype},
  author  = {Sandberg, Alexander and Hinostroza, Miguel and Lekkas, Anastasios M.},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3683312},
}
```

> **Note:** This is the codebase used in the paper and master’s thesis. A modernized version is currently in development on [`v2`](../../tree/v2).

The repository includes 
1. project_mAXAI - the project that run the project mAXAI code, including training, evaluating & deploying/testing for DRL and SHAP based exblainability dashboard for XAI
2. local_env - where collected data is used to analyse and generate plots and testing scripts are made
3. data - the environments, trained models and their data, including evaluations, monitorings and runs
