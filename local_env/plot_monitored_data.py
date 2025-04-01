##### Plot figures
######

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


scale = 1   # Change the scale 0.7/0.4

# Import
#monitor_data = pd.read_csv('models/MilliampereRos4Thrusters-v0.monitor copy.csv', skiprows=1)  # Skip the first row (header)

monitor_data = pd.read_csv('../data/models/training_20250328_131443/logs/monitor.csv',skiprows=1)  # Skip the first row (header)

r = monitor_data['r']
e = np.arange(len(r))
t = monitor_data['t']
l = monitor_data['l']


colors = ['green' if val >= 0 else 'red' for val in r]

# Plot

fig, ax = plt.subplots()

ax.bar(e,r, color=colors)
ax.axhline(y=1000, color='black', linestyle='--', label=f"Maximum Reward (y={1000})")
ax.legend()
ax.set_xlabel(r'Episode', fontsize=14*scale)
ax.set_ylabel(r'Episode Reward', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)

plt.tight_layout()
plt.savefig('plots/drl/episode_reward/episode_plot_2_r.pdf', format='pdf', dpi=1200)
plt.close(fig)


fig, ax = plt.subplots()

ax.plot(t,r, color='green')
ax.axhline(y=1000, color='black', linestyle='--', label=f"Maximum Reward (y={1000})")
ax.legend()
ax.set_xlabel(r'Time [s]', fontsize=14*scale)
ax.set_ylabel(r'Episode Reward', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)

plt.tight_layout()
plt.savefig('plots/drl/episode_reward/episode_plot_2_r2.pdf', format='pdf', dpi=1200)
plt.close(fig)

# Plot

fig, ax = plt.subplots()

ax.bar(e,l, color='blue')
ax.axhline(y=1000, color='black', linestyle='--', label=f"Maximum Length (y={1000})")
ax.legend()
ax.set_xlabel(r'Episode', fontsize=14*scale)
ax.set_ylabel(r'Episode Length $[\Delta t]$', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)

plt.tight_layout()
plt.savefig('plots/drl/episode_reward/episode_plot_2_l.pdf', format='pdf', dpi=1200)
plt.close(fig)