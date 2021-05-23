import os
import matplotlib.pyplot as plt


def plot_results(vertices_batch, filename, *, output_dir='plots', number_of_objects=4):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(20, 8))
    number_of_objects = min(vertices_batch.shape[0], number_of_objects)
    for i in range(number_of_objects):
        ax = fig.add_subplot(1, number_of_objects, i + 1,  projection='3d')
        x = vertices_batch[i, :, 0]
        y = vertices_batch[i, :, 1]
        z = vertices_batch[i, :, 2]
        ax.scatter(x, y, z)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

