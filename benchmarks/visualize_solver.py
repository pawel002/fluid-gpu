import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from fluidgpu import solver_steps

def ensure_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def create_gif(frames_dir, output_filename, duration=50):
    print(f"Generating GIF: {output_filename}...")
    frames = []
    imgs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    
    if not imgs:
        print("No frames found!")
        return

    for i in imgs:
        frames.append(Image.open(i))

    if frames:
        frames[0].save(
            output_filename,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=0
        )
        print(f"Saved to {output_filename}")

def run_visualization(scenario_name, u, v, nu, dt, dx, dy, total_steps, steps_per_frame):
    print(f"--- Starting Scenario: {scenario_name} ---")
    
    frames_dir = f"frames_3d_{scenario_name}"
    ensure_dir(frames_dir)
    
    ny, nx = u.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    mag = np.sqrt(u**2 + v**2)
    z_lims = {
        'mag': (0, mag.max() * 1.1),
        'u': (u.min() * 1.2 if u.min() < 0 else -0.2, u.max() * 1.2),
        'v': (v.min() * 1.2 if v.min() < 0 else -0.2, v.max() * 1.2)
    }

    fig = plt.figure(figsize=(20, 6), dpi=80)
    num_frames = total_steps // steps_per_frame
    
    for frame in range(num_frames):
        plt.clf()
        
        mag = np.sqrt(u**2 + v**2)
        
        # plot magnitude
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, mag, cmap='inferno', 
                               linewidth=0, antialiased=False, vmin=z_lims['mag'][0], vmax=z_lims['mag'][1])
        ax1.set_title("Velocity Magnitude $|u|$")
        ax1.set_zlim(z_lims['mag'])
        ax1.view_init(elev=35, azim=-45)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # plot u 
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u, cmap='coolwarm', 
                               linewidth=0, antialiased=False, vmin=z_lims['u'][0], vmax=z_lims['u'][1])
        ax2.set_title("U Component (Horizontal)")
        ax2.set_zlim(z_lims['u'])
        ax2.view_init(elev=35, azim=-45)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        # plot v
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        surf3 = ax3.plot_surface(X, Y, v, cmap='coolwarm', 
                               linewidth=0, antialiased=False, vmin=z_lims['v'][0], vmax=z_lims['v'][1])
        ax3.set_title("V Component (Vertical)")
        ax3.set_zlim(z_lims['v'])
        ax3.view_init(elev=35, azim=-45)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')

        time_sim = frame * steps_per_frame * dt
        plt.suptitle(f"{scenario_name} | Step: {frame * steps_per_frame} | t={time_sim:.3f}s", fontsize=16)

        plt.savefig(f"{frames_dir}/frame_{frame:04d}.png", bbox_inches='tight', facecolor='white')
        
        if frame % 10 == 0:
            print(f"Rendered frame {frame}/{num_frames}")

        solver_steps(
            u=u,
            v=v,
            nu=float(nu),
            dt=float(dt),
            dx=float(dx),
            dy=float(dy),
            steps=int(steps_per_frame),
        )

    plt.close()
    create_gif(frames_dir, f"{scenario_name}_3D.gif", duration=50)


def main():
    nx, ny = 128, 128  
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt = 0.0005
    
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")

    print("\nInitializing Scenario 1...")
    u1 = np.zeros((ny, nx), dtype=np.float32)
    v1 = np.zeros((ny, nx), dtype=np.float32)
    
    u1 += np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01).astype(np.float32)
    v1 += 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02).astype(np.float32)

    run_visualization(
        scenario_name="Gaussian_Diffusion",
        u=u1, v=v1, nu=0.005, dt=dt, dx=dx, dy=dy,
        total_steps=1500,
        steps_per_frame=15
    )

    print("\nInitializing Scenario 2...")
    u2 = np.zeros((ny, nx), dtype=np.float32)
    v2 = np.zeros((ny, nx), dtype=np.float32)

    blob1 = np.exp(-((X - 0.3) ** 2 + (Y - 0.5) ** 2) / 0.01)
    u2 += 1.0 * blob1
    
    blob2 = np.exp(-((X - 0.7) ** 2 + (Y - 0.5) ** 2) / 0.01)
    u2 -= 1.0 * blob2 

    run_visualization(
        scenario_name="Collision",
        u=u2.astype(np.float32), 
        v=v2.astype(np.float32), 
        nu=0.002, 
        dt=dt, dx=dx, dy=dy,
        total_steps=2000,
        steps_per_frame=15
    )

if __name__ == "__main__":
    main()