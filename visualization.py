# !/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import numpy as np


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_skeleton_connections():
    connections = [
        (0, 1), (0, 2),
        (1, 3), (2, 4),
        (0, 5), (0, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    return connections


def calculate_overall_limits(json_data, margin=0.2):
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for annotation in json_data['annotations']:
        keypoints = np.array(annotation['keypoints'])
        x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
        x_min, x_max = min(x_min, np.min(x)), max(x_max, np.max(x))
        y_min, y_max = min(y_min, np.min(y)), max(y_max, np.max(y))
        z_min, z_max = min(z_min, np.min(z)), max(z_max, np.max(z))

    return (x_min - margin, x_max + margin), (y_min - margin, y_max + margin), (z_min - margin, z_max + margin)


def plot_keypoints_3d(keypoints, connections=None, ax=None, scatter_kwargs=None, line_kwargs=None, limits=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if scatter_kwargs is None:
        scatter_kwargs = {'c': 'r', 's': 50}

    if line_kwargs is None:
        line_kwargs = {'c': 'b'}

    keypoints = np.array(keypoints)
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    z = keypoints[:, 2]

    ax.scatter(x, y, z, **scatter_kwargs)

    if connections is not None:
        for connection in connections:
            start, end = connection
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            x_vals = [keypoints[start][0], keypoints[end][0]]
            y_vals = [keypoints[start][1], keypoints[end][1]]
            z_vals = [keypoints[start][2], keypoints[end][2]]
            ax.plot(x_vals, y_vals, z_vals, **line_kwargs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if limits:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])

    return ax


def visualize_keypoints(json_data, output_dir='frames', use_connections=True):
    os.makedirs(output_dir, exist_ok=True)
    annotations = json_data['annotations']
    connections = get_skeleton_connections() if use_connections else None

    limits = calculate_overall_limits(json_data)

    for annotation in annotations:
        frame_index = annotation['frame_index']
        keypoints = annotation['keypoints']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        plot_keypoints_3d(keypoints, connections=connections, ax=ax, limits=limits)

        plt.title(f'Frame {frame_index}')
        plt.tight_layout()

        frame_path = os.path.join(output_dir, f'frame_{frame_index:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig)
        print(f'Saved {frame_path}')


def create_video_from_frames(frame_dir='frames', output_video='keypoints_visualization.mp4', fps=10):

    images = []
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    for filename in frame_files:
        filepath = os.path.join(frame_dir, filename)
        img = imageio.imread(filepath)
        if img.shape[2] == 4:
            img = img[..., :3]
        images.append(img)

    codec = 'libx264' if output_video.endswith('.mp4') else None

    imageio.mimwrite(output_video, images, fps=fps, codec=codec)
    print(f'Video saved as {output_video}')


def main():
    # json_path = './data/assistive_furniture/kinetics_val/0.json'
    json_path = '../pose-classification/data_preprocess/frame_64/kinetics_val/100.json'

    data = load_json(json_path)

    visualize_keypoints(data, output_dir='frames', use_connections=True)

    create_video_from_frames(frame_dir='frames', output_video='keypoints_visualization.mp4', fps=10)


if __name__ == '__main__':
    main()
