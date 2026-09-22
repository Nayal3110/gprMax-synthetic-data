import pyvista as pv
import numpy as np


def main():
    delta = 1.0
    plotter = pv.Plotter()
    arrow_scale = 0.2

    corners = np.array([
        [0, 0, 0],
        [delta, 0, 0],
        [delta, delta, 0],
        [0, delta, 0],
        [0, 0, -delta],
        [delta, 0, -delta],
        [delta, delta, -delta],
        [0, delta, -delta],
    ])

    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7),
    ]

    mid_corners = np.array([
        [delta/2, 0, 0],
        [delta/2, 0, -delta],
        [0, 0, -delta/2],
        [delta, 0, -delta/2],

        [0, delta/2, 0],
        [delta, delta/2, 0],
        [delta/2, 0, 0],
        [delta/2, delta, 0],

        [delta, delta/2, 0],
        [delta, delta/2, -delta],
        [delta, 0, -delta/2],
        [delta, delta, -delta/2],

        [0, delta/2, 0],
        [0, delta/2, -delta],
        [0, 0, -delta/2],
        [0, delta, -delta/2],

        [0, delta/2, -delta],
        [delta, delta/2, -delta],
        [delta/2, 0, -delta],
        [delta/2, delta, -delta],

        [0, delta, -delta/2],
        [delta, delta, -delta/2],
        [delta/2, delta, 0],
        [delta/2, delta, -delta],
    ])

    edges_mid = [
        (0,1), (2,3),
        (4,5), (6,7),
        (8,9), (10,11),
        (12,13), (14,15),
        (16,17), (18,19),
        (20,21), (22,23),
    ]

    offset = np.array([-delta/2, -delta/2, delta/2])
    s_offset = np.array([0, 0, delta])
    new_corners = corners + offset
    s_corners = corners + s_offset
    new_mid_corners = mid_corners + offset
    s_mid_corners = mid_corners + s_offset

    for e in edges:
        plotter.add_mesh(pv.Line(corners[e[0]], corners[e[1]]), color='blue', line_width=2)
        plotter.add_mesh(pv.Line(new_corners[e[0]], new_corners[e[1]]), color='purple', line_width=2)
        plotter.add_mesh(pv.Line(s_corners[e[0]], s_corners[e[1]]), color='orange', line_width=2)

    for e in edges_mid:
        plotter.add_mesh(pv.Line(mid_corners[e[0]], mid_corners[e[1]]), color='blue', line_width=2)
        plotter.add_mesh(pv.Line(new_mid_corners[e[0]], new_mid_corners[e[1]]), color='purple', line_width=2)
        plotter.add_mesh(pv.Line(s_mid_corners[e[0]], s_mid_corners[e[1]]), color='orange', line_width=2)

    H_x_pos = np.array([
        [0, delta/2, -delta/2], [delta, delta/2, -delta/2],
    ])
    H_y_pos = np.array([
        [delta/2, 0, -delta/2], [delta/2, delta, -delta/2],
    ])
    H_z_pos = np.array([
        [delta/2, delta/2, 0], [delta/2, delta/2, -delta],
    ])

    H_x_dirs = np.array([[1, 0, 0]] * 2)
    H_y_dirs = np.array([[0, 1, 0]] * 2)
    H_z_dirs = np.array([[0, 0, 1]] * 2)

    E_new_pos = np.vstack([H_x_pos, H_y_pos, H_z_pos]) + offset
    E_dirs = np.vstack([H_x_dirs, H_y_dirs, H_z_dirs])
    labels_E = ['E_x'] * 2 + ['E_y'] * 2 + ['E_z'] * 2

    H_s_pos = np.vstack([H_x_pos, H_y_pos, H_z_pos]) + s_offset

    for pos, dir_vec, label in zip(
        np.vstack([H_x_pos, H_y_pos, H_z_pos]),
        np.vstack([H_x_dirs, H_y_dirs, H_z_dirs]),
        ['H_x'] * 2 + ['H_y'] * 2 + ['H_z'] * 2,
    ):
        arrow = pv.Arrow(start=pos, direction=dir_vec, scale=arrow_scale)
        plotter.add_mesh(arrow, color='green')
        plotter.add_point_labels([pos], [label], font_size=12, text_color='black')

    for pos, dir_vec, label in zip(E_new_pos, E_dirs, labels_E):
        arrow = pv.Arrow(start=pos, direction=dir_vec, scale=arrow_scale)
        plotter.add_mesh(arrow, color='red')
        plotter.add_point_labels([pos], [label], font_size=12, text_color='black')

    for pos, dir_vec, label in zip(
        H_s_pos,
        np.vstack([H_x_dirs, H_y_dirs, H_z_dirs]),
        ['H_x'] * 2 + ['H_y'] * 2 + ['H_z'] * 2,
    ):
        arrow = pv.Arrow(start=pos, direction=dir_vec, scale=arrow_scale)
        plotter.add_mesh(arrow, color='green')
        plotter.add_point_labels([pos], [label], font_size=12, text_color='black')

    E_x_pos = np.array([
        [delta/2, 0, 0], [delta/2, delta, 0], [delta/2, 0, -delta], [delta/2, delta, -delta],
    ])
    E_y_pos = np.array([
        [0, delta/2, 0], [delta, delta/2, 0], [0, delta/2, -delta], [delta, delta/2, -delta],
    ])
    E_z_pos = np.array([
        [0, 0, -delta/2], [delta, 0, -delta/2], [0, delta, -delta/2], [delta, delta, -delta/2],
    ])

    E_x_dirs = np.array([[1, 0, 0]] * 4)
    E_y_dirs = np.array([[0, 1, 0]] * 4)
    E_z_dirs = np.array([[0, 0, 1]] * 4)

    H_new_pos = np.vstack([E_x_pos, E_y_pos, E_z_pos]) + offset
    H_dirs = np.vstack([E_x_dirs, E_y_dirs, E_z_dirs])
    labels_H = ['H_x'] * 4 + ['H_y'] * 4 + ['H_z'] * 4

    E_s_pos = np.vstack([E_x_pos, E_y_pos, E_z_pos]) + s_offset

    for pos, dir_vec, label in zip(
        np.vstack([E_x_pos, E_y_pos, E_z_pos]),
        np.vstack([E_x_dirs, E_y_dirs, E_z_dirs]),
        ['E_x'] * 4 + ['E_y'] * 4 + ['E_z'] * 4,
    ):
        arrow = pv.Arrow(start=pos, direction=dir_vec, scale=arrow_scale)
        plotter.add_mesh(arrow, color='red')
        plotter.add_point_labels([pos], [label], font_size=10, text_color='black')

    for pos, dir_vec, label in zip(H_new_pos, H_dirs, labels_H):
        arrow = pv.Arrow(start=pos, direction=dir_vec, scale=arrow_scale)
        plotter.add_mesh(arrow, color='green')
        plotter.add_point_labels([pos], [label], font_size=10, text_color='black')

    for pos, dir_vec, label in zip(
        E_s_pos,
        np.vstack([E_x_dirs, E_y_dirs, E_z_dirs]),
        ['E_x'] * 4 + ['E_y'] * 4 + ['E_z'] * 4,
    ):
        arrow = pv.Arrow(start=pos, direction=dir_vec, scale=arrow_scale)
        plotter.add_mesh(arrow, color='red')
        plotter.add_point_labels([pos], [label], font_size=10, text_color='black')

    plotter.show_axes()
    plotter.show()


if __name__ == "__main__":
    main()
