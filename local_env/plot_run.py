import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad
from matplotlib.lines import Line2D
import time

df = pd.read_csv("../data/runs/sim/data_test_spline_20250520_161615.csv")

scale = 10/6.4
render_start = 0
render_end = 200*5
run_number = 3

x = df['x']*10
y = df['y']*10
psi = df['psi']*180
u = df['u']*3.24
v = df['v']*3.24
r = df['r']*112.6
n1x = df['n1x']*900
n1y = df['n1y']*900
n2x = df['n2x']*900
n2y = df['n2y']*900
n3x = df['n3x']*900
n3y = df['n3y']*900
n4x = df['n4x']*900
n4y = df['n4y']*900
n1 = df['n1d']
alpha1 = df['alpha1d']
n2 = df['n2d']
alpha2 = df['alpha2d']
n3 = df['n3d']
alpha3 = df['alpha3d']
n4 = df['n4d']
alpha4 = df['alpha4d']

dt = np.arange(len(x))


def mark_episode(render_start, render_end, ax):
    # Assume render_start = 600, render_end = 2600 and you want 200-step intervals.
    interval = 200
    flag = True  # use this to alternate colors

    for start in range(render_start, render_end, interval):
        end = start + interval
        # Check if the ending goes beyond render_end, if so, truncate:
        if end > render_end:
            end = render_end
            
        if flag:
            # Add a gray vertical span for these 200 time steps
            ax.axvspan(start, end, facecolor='gray', alpha=0.2)
        else:
            # Else leave it white or add another color if you like
            # ax.axvspan(start, end, facecolor='white', alpha=1.0)  # Typically unnecessary if background is white by default
            pass
        
        # Flip the flag for the next iteration
        flag = not flag



# X
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"Set point (y={0})")
ax.plot(dt[render_start:render_end],x[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\tilde{x}^b_t$ [m]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)

plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_x.pdf', format='pdf', dpi=1200)
plt.close(fig)

# Y
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"Set point (y={0})")
ax.plot(dt[render_start:render_end],y[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\tilde{y}^b_t$ [m]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_y.pdf', format='pdf', dpi=1200)
plt.close(fig)

# psi
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"Set point (y={0})")
ax.plot(dt[render_start:render_end],psi[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\tilde{\psi}_t$ [$^\circ$]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_psi.pdf', format='pdf', dpi=1200)
plt.close(fig)

# U
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"Set point (y={0})")
ax.plot(dt[render_start:render_end],u[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\hat{u}_t$ [m/s]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_u.pdf', format='pdf', dpi=1200)
plt.close(fig)

# V
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"Set point (y={0})")
ax.plot(dt[render_start:render_end],v[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\hat{v}_t$ [m/s]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_v.pdf', format='pdf', dpi=1200)
plt.close(fig)

# R
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5*scale,label=f"Set point (y={0})")
ax.plot(dt[render_start:render_end],r[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\hat{r}_t$ [$^\circ$/s]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_r.pdf', format='pdf', dpi=1200)
plt.close(fig)

# N1x
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
ax.plot(dt[render_start:render_end],n1[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$n_{1,d,t-1}$ [RPM]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_n1.pdf', format='pdf', dpi=1200)
plt.close(fig)

# N2
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', label=f"(y={0})")
ax.plot(dt[render_start:render_end],n2[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$n_{d_2,t-1}$ [RPM]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_n2.pdf', format='pdf', dpi=1200)
plt.close(fig)

# N3
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
ax.plot(dt[render_start:render_end],n3[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$n_{d_3,t-1}$ [RPM]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_n3.pdf', format='pdf', dpi=1200)
plt.close(fig)

# N4
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
ax.plot(dt[render_start:render_end],n4[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$n_{d_4,t-1}$ [RPM]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_n4.pdf', format='pdf', dpi=1200)
plt.close(fig)

# ALPHA1
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
ax.plot(dt[render_start:render_end],alpha1[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\alpha_{d_1,t-1}$ [$^\circ$]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_alpha1.pdf', format='pdf', dpi=1200)
plt.close(fig)

# ALPHA2
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
ax.plot(dt[render_start:render_end],alpha2[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\alpha_{d_2,t-1}$ [$^\circ$]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_alpha2.pdf', format='pdf', dpi=1200)
plt.close(fig)


# ALPHA3
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
ax.plot(dt[render_start:render_end],alpha3[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\alpha_{d_3,t-1}$ [$^\circ$]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_alpha3.pdf', format='pdf', dpi=1200)
plt.close(fig)


# ALPHA4
fig, ax = plt.subplots(figsize=(10,4.8))

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5*scale,label=f"(y={0})")
ax.plot(dt[render_start:render_end],alpha4[render_start:render_end], linewidth=1.5*scale, color='black')
ax.legend(fontsize=10*scale)
ax.set_xlabel(r'time step [dt]', fontsize=14*scale)
ax.set_ylabel(r'$\alpha_{d_4,t-1}$ [$^\circ$]', fontsize=14*scale)
ax.tick_params(axis='x', labelsize=10*scale)
ax.tick_params(axis='y', labelsize=10*scale)
mark_episode(render_start, render_end, ax)


plt.tight_layout()
plt.savefig(f'plots/drl/runs/{run_number}_alpha4.pdf', format='pdf', dpi=1200)
plt.close(fig)


def _rotate_body2ned(psi):
    psi = np.deg2rad(psi)
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c,-s],[s,c]])

xy = np.array([list(x),list(y)])
print(xy.shape)
print(xy.T.shape)
xy = [-_rotate_body2ned(-p)@xy[:,i] for i,p in enumerate(psi)]
xy = np.array(xy)
print(xy.shape)
x=xy[:,0]
y=xy[:,1]


for a in range(5):
    offset = a*200*0
    k = 10*0

    fig, ax = plt.subplots()
    print(xy)
    ax.axhline(y=0, color='gray', linestyle='--', label=f"(y={0})")
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.plot(y[render_start+k+offset:render_start+offset+200],x[render_start+k+offset:render_start+offset+200], color='red')
    ax.set_xlabel(r'$y^n$', fontsize=14)
    ax.set_ylabel(r'$x^n$', fontsize=14)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim((-7,7))
    ax.set_ylim((-7,7))
    ax.set_aspect('equal', adjustable='datalim')

    my_range = [render_start + k + offset, render_end-1]

    # Add rectangles every 20 time steps
    for t in my_range:
        print(len(y))
        # Define rectangle properties
        rect_x = y[t]  # Use the `y` coordinate for the rectangle center
        rect_y = x[t]  # Use the `x` coordinate for the rectangle center
        width, height = 5.06, 2.86  # Dimensions of the rectangle
        angle = 90-psi[t]  # Rotation angle in degrees

        # Create the rectangle
        rect = Rectangle((rect_x - width / 2, rect_y - height / 2), width, height,
                         angle=0, color='blue', alpha=0.6)

        # Rotate the rectangle around its center
        trans = transforms.Affine2D().rotate_deg_around(rect_x, rect_y, angle) + ax.transData
        rect.set_transform(trans)

        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Add vectors every 20 time steps
    for t in range(render_start + k + offset, render_start + offset + 200, 10):
        # Define vector properties
        vector_x = y[t]  # Starting point of the vector (y-coordinate)
        vector_y = x[t]  # Starting point of the vector (x-coordinate)
        angle = np.deg2rad(90-psi[t])  # Convert angle to radians for vector calculation

        # Vector components
        dx = np.cos(angle)  # x-component of the vector
        dy = np.sin(angle)  # y-component of the vector

        # Scale the vector length (optional)
        scale_factor = 0.5*2
        dx *= scale_factor
        dy *= scale_factor

        # Plot the vector
        ax.quiver(vector_x, vector_y, dx, dy, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'plots/drl/runs/{run_number}_xy{a}.pdf', format='pdf', dpi=1200)
    plt.close(fig)



def drawCirc(ax,radius,centX,centY,angle_,theta1_,theta2_,color_='black'):
    #========Line
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=theta1_, theta2=theta2_,linestyle='-',lw=3,color=color_)
    ax.add_patch(arc)
    legend_element = Line2D([0], [0], color=color_, lw=3, linestyle='-', label=r'Angular Velocity [$^\circ$/s]')
    
    startX=centX+(radius/2)*np.cos(rad(angle_))
    startY=centY+(radius/2)*np.sin(rad(angle_))
    ax.plot(startX, startY, 'ko', markersize=3)


    if theta1_ == 0:
        ang = theta2_ 
    
        #========Create the arrow head
        endX=centX+(radius/2)*np.cos(rad(ang+angle_)) #Do trig to determine end position
        endY=centY+(radius/2)*np.sin(rad(ang+angle_))
        ax.add_patch(                    #Create triangle as arrow head
            RegularPolygon(
                (endX, endY),            # (x,y)
                3,                       # number of vertices
                radius=radius/9,                # radius
                orientation=rad(angle_+ang),     # orientation
                color=color_
            )
        )
    else:
        ang = theta1_

        #========Create the arrow head
        endX=centX+(radius/2)*np.cos(rad(ang+angle_)) #Do trig to determine end position
        endY=centY+(radius/2)*np.sin(rad(ang+angle_))
        ax.add_patch(                    #Create triangle as arrow head
            RegularPolygon(
                (endX, endY),            # (x,y)
                3,                       # number of vertices
                radius=radius/9,                # radius
                orientation=rad(angle_+ang+180),     # orientation
                color=color_
            )
        )

    return legend_element
    #ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort


for a in range(20):
    offset = a*201*0
    k = 10*a

    fig, ax = plt.subplots()
    print(xy)
    ax.axhline(y=0, color='gray', linestyle='--', label=f"(y={0})")
    ax.axvline(x=0, color='gray', linestyle='--')
    #ax.plot(y[render_start+k+offset:render_start+offset+201],x[render_start+k+offset:render_start+offset+201], color='red')
    ax.set_xlabel(r'$y^n$', fontsize=14)
    ax.set_ylabel(r'$x^n$', fontsize=14)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim((-2,10))
    ax.set_ylim((-6,4))
    ax.set_aspect('equal', adjustable='datalim')
    
    legend_handles = []

    my_range = [render_start + k + offset, render_end]
    
    t=render_start + k + offset
    # Add rectangles every 20 time steps
    # Define rectangle properties
    rect_x = y[t]  # Use the `y` coordinate for the rectangle center
    rect_y = x[t]  # Use the `x` coordinate for the rectangle center
    width, height = 5.06, 2.86  # Dimensions of the rectangle
    angle = 90-psi[t]  # Rotation angle in degrees


    rect_x_list = [rect_x,0]
    rect_y_list = [rect_y,0]
    angle_list = [angle,90]

    color = ['blue','red','blue','none']

    for i in range(2):
        # Create the rectangle
        rect = Rectangle((rect_x_list[i] - width / 2, rect_y_list[i] - height / 2), width, height,
                        angle=0, edgecolor=color[i], facecolor=color[i+2], alpha=0.6)

        # Rotate the rectangle around its center
        trans = transforms.Affine2D().rotate_deg_around(rect_x_list[i], rect_y_list[i], angle_list[i]) + ax.transData
        rect.set_transform(trans)

        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Define thruster positions relative to the rectangle's center
    Lx = 1.8
    Ly = 0.8
    thruster_positions = np.array([[Lx, Ly],
                                    [Lx, -Ly],
                                    [-Lx, Ly],
                                    [-Lx, -Ly]])


    rect_x = y[t]
    rect_y = x[t]
    angle = np.deg2rad(90 - psi[t])  # Convert angle to radians for rotation
    

    # Rotation matrix for the rectangle's orientation
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Compute global coordinates of the thruster positions
    global_thruster_positions = rotation_matrix @ thruster_positions.T
    global_thruster_positions[0, :] += rect_x
    global_thruster_positions[1, :] += rect_y
    ax.plot(rect_x, rect_y, 'ko', markersize=3, label='Point Label')

    # Debugging output
    print(f"Time step {t}: Rectangle Center ({rect_x}, {rect_y})")
    print(f"Global Thruster Positions:\n{global_thruster_positions}")

    delta = [n1,n2,n3,n4,alpha1,alpha2,alpha3,alpha4]
    # Plot vectors originating from thruster positions
    for i in range(4):
        thruster_x = global_thruster_positions[0, i]
        thruster_y = global_thruster_positions[1, i]

        # Define vector components (customize these as needed)
        vector_angle = angle-np.deg2rad(delta[i+4][t])  # Assuming the vector points along the rectangle's orientation
        dx = np.cos(vector_angle) * 2*np.pi/1200 * delta[i][t]  # Scale vector length as needed
        dy = np.sin(vector_angle) * 2*np.pi/1200 * delta[i][t] 
        print(f"alpha_{i}: {delta[i+4][t]}, n_{i}: {delta[i][t]}")
        
        ax.plot(thruster_x, thruster_y, 'ko', markersize=3, label='Point Label')
        
        # Plot the vector
        if i == 0:
            thrust_handle = ax.quiver(thruster_x, thruster_y, dx, dy, angles='xy', scale_units='xy', scale=1, color="lime",alpha=1, label='Thrust [RPM]')
        else:
            ax.quiver(thruster_x, thruster_y, dx, dy, angles='xy', scale_units='xy', scale=1, color="lime",alpha=1)
    legend_handles.append(thrust_handle)
    # Linear Speed Vector
    print(f"u: {u[t]}, v: {v[t]}, r: {r[t]}" )
    linear_speed = np.hypot(u[t], v[t])  # Compute linear speed magnitude
    dx = u[t]*np.cos(np.deg2rad(-psi[t]))-v[t]*np.sin(np.deg2rad(-psi[t]))  # x-component of the linear velocity
    dy = u[t]*np.sin(np.deg2rad(-psi[t]))+v[t]*np.cos(np.deg2rad(-psi[t]))  # y-component of the linear velocity
    #dx = v[t]
    #dy = u[t]
    scale_factor = 2*np.pi/3.5  # Adjust this for arrow length scaling
    linear_handle = ax.quiver(rect_x, rect_y, -dx * scale_factor, -dy * scale_factor, angles='xy', scale_units='xy',
            scale=1, color='orange', alpha=0.8, label='Linear Speed [m/s]')
    legend_handles.append(linear_handle)
    
    angle = 360*r[t]/120
    if angle > 0:
        circ_handle = drawCirc(ax,1,rect_x,rect_y,90-psi[t],-angle,0,color_='purple')
    else:
        circ_handle = drawCirc(ax,1,rect_x,rect_y,90-psi[t],0,360-angle,color_='purple')
    
    legend_handles.append(circ_handle)

    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(f'plots/drl/runs/{run_number}_xy2{a}.pdf', format='pdf', dpi=1200)
    plt.close(fig)