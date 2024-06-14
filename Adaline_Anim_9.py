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

t_step=np.arange(1,L)

w1 = 2 * np.pi * 50  # Fundamental Frequency
Va = va_dc * np.exp(-beta * t) + \
     V1 * np.sin(w1 * t) + \
     V3 * np.sin(3 * w1 * t + 3 * np.pi / 180) + \
     V5 * np.sin(5 * w1 * t + 5 * np.pi / 180) + \
     V7 * np.sin(7 * w1 * t + 2 * np.pi / 180)  # Main Signal Creation

# Inputs and Weights Vectors Initialization
x = np.ones(Nx)
w = np.ones(Nx)

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    colors_list=[
        ['b','b','g','g','purple','purple','orange','orange','w'], # weights
        ['c'], # sigma
        ['w'] #output
    ]

    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.*1.8,
                                color='k', ec=colors_list[n][m], zorder=4,lw=2)
            ax.add_artist(circle)
            if n==0:
                ax.annotate('d', xy=((n*h_spacing + left)+0.07, (layer_top - m*v_spacing)+0.07), xytext=(n*h_spacing + left, layer_top - m*v_spacing),
                         arrowprops=dict(width=2,facecolor=colors_list[n][m], edgecolor='none', shrink=0.001))
            

            
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                #line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],[layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c=colors_list[n][m],lw=2)
                #ax.add_artist(line)
                ax.annotate('d', xytext=(n*h_spacing + left,layer_top_a - m*v_spacing), xy=((n + 1)*h_spacing + left-0.04,layer_top_b - o*v_spacing),
                         arrowprops=dict(width=2,facecolor=colors_list[n][m], edgecolor='none', shrink=0.001),zorder=0)





fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor('k')
ax = fig.gca()
ax.axis('off')

draw_neural_net(ax, .1, .9, .1, .9, [Nx, 1, 1])

# Adding labels
inputs = ['$Sin(\omega_1t)$', '$Cos(\omega_1t)$', '$Sin(3\omega_1t)$', '$Cos(3\omega_1t)$', '$Sin(5\omega_1t)$', '$Cos(5\omega_1t)$', '$Sin(7\omega_1t)$', '$Cos(7\omega_1t)$', '$1$']
for i, label in enumerate(inputs):
    ax.text(0.001, 0.85 - (i * 0.09), label, ha='right',color='w',fontsize=15,weight='bold')
    ax.annotate('', xy=(0.05, 0.85 - (i * 0.089)), xytext=(0.01, 0.85 - (i * 0.089)), arrowprops=dict(width=1.5,facecolor='w', edgecolor='none', shrink=0.01))

ax.text(00, 0.95,r'$\overrightarrow{X}_{t}$', ha='center',color='w',fontsize=15,weight='bold')


weights = [ax.text(0.1, 0.85 - (i * 0.089), '', ha='center',color='yellow',weight='bold',fontstyle='italic',zorder=9) for i in range(Nx)]
output_text = ax.text(0.7, 0.52, '', ha='center',color='cyan',weight='bold',fontsize=13)
desired_output_text = ax.text(0.7, 0.85, '', ha='center',color='yellow',weight='bold',fontsize=13)
error_text = ax.text(0.68, 0.36, '', ha='left',color='r',weight='bold',fontsize=13)
time_text = ax.text(0.5, 0.95, '', ha='center',color='yellow',weight='bold',fontsize=15)

ax.text(0.519, 0.475, '$\Sigma$', ha='right',color='yellow',zorder=9,fontsize=30,weight='heavy',fontstyle='italic')

ax.text(0.86, 0.53, '+', ha='right',color='c',zorder=9,fontsize=20,weight='heavy',fontstyle='italic')
ax.text(0.95, 0.6, '_', ha='right',color='yellow',zorder=9,fontsize=20,weight='heavy',fontstyle='italic')

ax.annotate('d', xy=(0.9, 0.53), xytext=(0.9, 0.9), arrowprops=dict(width=2,facecolor='yellow', edgecolor='none', shrink=0.05))

ax.annotate('d', xy=(0.9, 0.1), xytext=(0.9, 0.53), arrowprops=dict(width=2,facecolor='r', edgecolor='none', shrink=0.05))
ax.text(0.9, 0.08, r'$\overrightarrow{W}_{t+1}=\overrightarrow{W}_t+\alpha*E_t*\frac{\overrightarrow{X}_t}{\overrightarrow{X}_t.\overrightarrow{X}_t}$', ha='center',color='w',zorder=9,fontsize=17)
ax.annotate('d', xy=(0.1, 0.05), xytext=(0.8, 0.05), arrowprops=dict(width=5,facecolor='yellowgreen', edgecolor='w', shrink=0.01))
ax.annotate('', xy=(0.1, 0.9), xytext=(0.1, 0.03), arrowprops=dict(width=5,facecolor='yellowgreen', edgecolor='w', shrink=0.01))


ax.text(0.125, 0.95,r'$\overrightarrow{W}_{t}$', ha='right',color='greenyellow',fontsize=15,weight='bold')

# Arrows for error and weight updates
'''
arrowprops = dict(facecolor='w', shrink=0.05)
ax.annotate('', xy=(0.75, 0.5), xytext=(0.75, 0.4), arrowprops=arrowprops)
ax.annotate('Error', xy=(0.85, 0.35), xytext=(0.9, 0.3), arrowprops=arrowprops)
for i in range(Nx):
    ax.annotate('', xy=(0.75, 0.5), xytext=(0.5, 0.9 - (i * 0.1)), arrowprops=arrowprops)
'''

# Initialization function to set up the background of each frame
def init():
    for w_text in weights:
        w_text.set_text('')
        output_text.set_text('')
        desired_output_text.set_text('')
        error_text.set_text('')
        time_text.set_text('')
    return weights + [output_text, desired_output_text, error_text,time_text]



# Animation function to update the data for each frame
def update(frame):
    global w, x

    x = np.array([np.sin(w1 * t[frame]), np.cos(w1 * t[frame]), np.sin(3 * w1 * t[frame]), 
                  np.cos(3 * w1 * t[frame]), np.sin(5 * w1 * t[frame]), np.cos(5 * w1 * t[frame]), 
                  np.sin(7 * w1 * t[frame]), np.cos(7 * w1 * t[frame]), 1])
    
    y = np.dot(w, x)  # Neuron's Output
    e = Va[frame] - y  # Error
    w += alpha * (e * x) / np.dot(x, x)  # Updating Weights using widrow-hoff Law
    
    for i, w_text in enumerate(weights):
        w_text.set_text(f'{w[i]:.2f}')
        output_text.set_text(f'$Output:$ {y:.2f}')
        desired_output_text.set_text(f'$Desired Output:$ {Va[frame]:.2f}')
        error_text.set_text(f'$Error:$ {e:.2f}')
        time_text.set_text(f'$t=${t_step[frame]}$($${t[frame]:.3f}$ $ms)$')
    
    return weights + [output_text, desired_output_text, error_text,time_text]

# Create the animation
ani = FuncAnimation(fig, update, frames=L, init_func=init, blit=True, interval=50, repeat=False)

# Display the animation
#plt.show()

# Save the animation as .mp4
writer = animation.writers['ffmpeg'](fps=30)
#ani.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-crf', '18'])
ani.save('neuron.mp4',writer=writer,dpi=180)
# Display the animation
#plt.show()
