import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

# Parameters    
tmax = 0.15  # Maximum time
fs = 2e3  # Sampling frequency
N = 7  # Maximum Order of Harmonics (Without Considering Even-Order Harmonics)
alpha = 0.4  # Learning Rate
va_dc = 60  # Trans. DC Amp.
beta = 35  # Trans. DC time cons. invers
V1 = 100  # Fundamental Harmonic Amplitude
V3 = 0  # 3rd Harmonic Amplitude
V5 = 30  # 5th Harmonic Amplitude
V7 = 15  # 7th Harmonic Amplitude

# Calculations
Nx = (N + 1) + 1  # Number of Inputs = Number of Weights
Ts = 1 / fs  # Sampling Period
t = np.arange(0, tmax + Ts, Ts)  # Time Vector
L = len(t)
w1 = 2 * np.pi * 50  # Fundamental Frequency
Va = va_dc * np.exp(-beta * t) + \
     V1 * np.sin(w1 * t) + \
     V3 * np.sin(3 * w1 * t + 3 * np.pi / 180) + \
     V5 * np.sin(5 * w1 * t + 5 * np.pi / 180) + \
     V7 * np.sin(7 * w1 * t + 2 * np.pi / 180)  # Main Signal Creation

# Inputs and Weights Vectors Initialization
x = np.ones(Nx)
w = np.ones(Nx)

# Adaptive Linear Neuron On-Line Learning
Va_made = np.zeros(L)
phi1, phi3, phi5, phi7 = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
r1, r3, r5, r7 = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
va_offset = np.zeros(L)

for k in range(L):
    x = np.array([np.sin(w1 * t[k]), np.cos(w1 * t[k]), np.sin(3 * w1 * t[k]), np.cos(3 * w1 * t[k]),
                  np.sin(5 * w1 * t[k]), np.cos(5 * w1 * t[k]), np.sin(7 * w1 * t[k]), np.cos(7 * w1 * t[k]),
                  1])
    y = np.dot(w, x)  # Neuron's Output
    e = Va[k] - y  # Error
    w += alpha * (e * x) / np.dot(x, x)  # Updating Weights using widrow-hoff Law

    # Extracting Amplitude and Phase of Harmonics from new weights' values
    phi1[k], r1[k] = np.arctan2(w[1], w[0]), np.hypot(w[1], w[0])
    phi3[k], r3[k] = np.arctan2(w[3], w[2]), np.hypot(w[3], w[2])
    phi5[k], r5[k] = np.arctan2(w[5], w[4]), np.hypot(w[5], w[4])
    phi7[k], r7[k] = np.arctan2(w[7], w[6]), np.hypot(w[7], w[6])
    va_offset[k] = w[8]

    # Creating Signal by New Values
    Va_made[k] = va_offset[k] + r1[k] * np.sin(w1 * t[k] + phi1[k]) + \
                 r3[k] * np.sin(3 * w1 * t[k] + phi3[k]) + \
                 r5[k] * np.sin(5 * w1 * t[k] + phi5[k]) + \
                 r7[k] * np.sin(7 * w1 * t[k] + phi7[k])

# Set up the figure and axis with a black background
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.patch.set_facecolor('black')
for ax in axs:
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    #ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    #ax.spines['right'].set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

axs[0].grid(visible=1,which='major',color='w', linestyle=':', linewidth=0.5)
axs[1].grid(visible=1,which='major',color='w', linestyle=':', linewidth=0.5)
#axs[2].spines['right'].set_color('white')
axs[2].grid(visible=1,which='major',color='w', linestyle=':', linewidth=0.5)
axs[3].grid(visible=1,which='major',color='w', linestyle=':', linewidth=0.5)

# Add mathematical expression of the main signal above the first box
axs[0].text(0.5, 1.4, r'$V(t) = A e^{-\beta t} + V_1 \sin(\omega_1 t) + V_3 \sin(3\omega_1 t + \phi_3) + V_5 \sin(5\omega_1 t + \phi_5) + V_7 \sin(7\omega_1 t + \phi_7)$',
            horizontalalignment='center',fontweight=100,  verticalalignment='bottom', fontsize=15, color='yellow', transform=axs[0].transAxes)
axs[0].text(0.5, 1.1, r'$A=60, \beta=35,V_1=100,V_3=0,V_5=20,V_7=15,\omega_1=2\pi50$',
            horizontalalignment='center', verticalalignment='bottom', fontsize=13, color='yellow', transform=axs[0].transAxes)
# Create line objects for each signal
lines = [
    axs[0].plot([], [], lw=2, color='yellow', label='$Original-Signal$')[0],
    axs[0].plot([], [], lw=2, color='cyan', label='$AdaLiNe-Output$')[0],
    axs[1].plot([], [], lw=2, color='red', label='$Error$')[0],
    axs[2].plot([], [], lw=2, color='blue', label='$V_1$')[0],
    axs[2].plot([], [], lw=2, color='green', label='$V_3$')[0],
    axs[2].plot([], [], lw=2, color='purple', label='$V_5$')[0],
    axs[2].plot([], [], lw=2, color='orange', label='$V_7$')[0],
    axs[3].plot([], [], lw=2, color='white', label='$V_{dc}$')[0]
]

# Set axis limits
axs[0].set_xlim(0, tmax * 1e3)
axs[0].set_ylim(np.min(Va) - 10, 199)#np.max(Va) + 40)
axs[1].set_xlim(0, tmax * 1e3)
axs[1].set_ylim(np.min(Va - Va_made) - 10, np.max(Va - Va_made) + 10)
axs[2].set_xlim(0, tmax * 1e3)
axs[2].set_ylim(-5,149)# np.max([r1,r3,r5,r7])+30)
axs[3].set_xlim(0, tmax * 1e3)
axs[3].set_ylim(np.min(va_offset) - 10, np.max(va_offset) + 10)
#
axs[2].set_yticks([V1,V3,V5,V7])
# Legends and labels
axs[0].set_ylabel("$Amplitude (V)$")
axs[0].legend(loc='upper right',labelcolor='white',frameon=False,ncol=2,fontsize='x-large')
axs[1].set_ylabel("$Amplitude (V)$")
axs[1].legend(loc='upper right',labelcolor='white',frameon=False,fontsize='x-large')
axs[2].set_ylabel("$Amplitude (V)$")
axs[2].legend(loc='upper right',labelcolor='white',frameon=False,ncol=4,fontsize='x-large')
axs[3].set_ylabel("$Amplitude (V)$")
axs[3].set_xlabel("$Time (ms)$")
axs[3].legend(loc='upper right',labelcolor='white',frameon=False,fontsize='x-large')

# Initialization function to set up the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Animation function to update the data for each frame
def update(frame):
    lines[0].set_data(t[:frame] * 1e3, Va[:frame])
    lines[1].set_data(t[:frame] * 1e3, Va_made[:frame])
    lines[2].set_data(t[:frame] * 1e3, Va[:frame] - Va_made[:frame])
    lines[3].set_data(t[:frame] * 1e3, r1[:frame])
    lines[4].set_data(t[:frame] * 1e3, r3[:frame])
    lines[5].set_data(t[:frame] * 1e3, r5[:frame])
    lines[6].set_data(t[:frame] * 1e3, r7[:frame])
    lines[7].set_data(t[:frame] * 1e3, va_offset[:frame])
    return lines

# Create the animation
ani = FuncAnimation(fig, update, frames=L, init_func=init, blit=True, interval=50,repeat=False)

# Display the animation
#plt.show()


# Save the animation as .mp4
writer = animation.writers['ffmpeg'](fps=30)
#ani.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-crf', '18'])
ani.save('alpha9.mp4',writer=writer,dpi=180)
# Display the animation
#plt.show()
