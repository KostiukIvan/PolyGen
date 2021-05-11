import os
import matplotlib.pyplot as plt


def plot_results(vertices, filename, *, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax.scatter(x, y, z)
    plt.savefig(os.path.join(output_dir, filename))
