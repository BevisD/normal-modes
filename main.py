import numpy as np
import matplotlib.pyplot as plt
import pygame
import tkinter as tk

WIDTH = 600
HEIGHT = 600
MASS_COLOR = (0, 0, 255)
SPRING_COLOR = (0, 0, 0)
TIME_STEP = 0.005


def calculate_normal_modes(T, V):
    eigenvalues, eigenvectors = np.linalg.eigh(V)
    omegas = np.sqrt(eigenvalues)
    return np.round(omegas, 6), np.round(eigenvectors, 6)


def calculate_coefficients(omegas, modes, equilib_coords,
                           initial_coords, initial_speeds):
    coeffs = np.array([(0, 0) for j in omegas], dtype=np.float64)
    offsets = np.subtract(initial_coords, equilib_coords)

    for j in range(len(omegas)):
        coeffs[j][1] = np.round(np.dot(offsets, modes[:, j]), 6)
        if omegas[j] == 0:
            coeffs[j][0] = np.round(np.dot(initial_speeds, modes[:, j]), 6)
        else:
            coeffs[j][0] = np.round(
                np.dot(initial_speeds, modes[:, j]) / omegas[j], 6)

    return coeffs


def calculate_positions(omegas, modes, coefficients, time):
    deltas = np.zeros(np.shape(omegas))
    for j in range(len(omegas)):
        if omegas[j] == 0:
            deltas += modes[:, j] * coefficients[j][0] * time + \
                      modes[:, j] * coefficients[j][1]
        else:
            deltas += modes[:, j] * coefficients[j][0] * np.sin(omegas[j] * time) + \
                      modes[:, j] * coefficients[j][1] * np.cos(omegas[j] * time)
    return deltas


def calculate_velocities(omegas, modes, coefficients, time):
    velocities = np.zeros(np.shape(omegas))
    for j in range(len(omegas)):
        if omegas[j] == 0:
            velocities += modes[:, j] * coefficients[j][0] * time
        else:
            velocities += modes[:, j] * coefficients[j][0] * \
                          np.cos(omegas[j] * time) * omegas[j] - \
                          modes[:, j] * coefficients[j][1] * \
                          np.sin(omegas[j] * time) * omegas[j]
    return velocities


def quit_callback():
    global ended
    ended = True


def togglePause():
    global paused
    paused = not paused


if __name__ == "__main__":
    # Coordinates given as fractions of window width
    equilib_coords = [0.25, 0.5, 0.75]
    initial_coords = [0.25, 0.5, 0.75]
    initial_speeds = [0.02, -0.01, -0.01]

    # Kinetic Energy Tensor
    T = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    # Potential Energy Tensor
    V = np.array([[1, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])

    # Calculate Normal Modes of the System
    omegas, modes = calculate_normal_modes(T, V)

    # Calculate coefficients of each normal mode
    coefficients = calculate_coefficients(omegas, modes, equilib_coords,
                                          initial_coords, initial_speeds)

    t = 0

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    paused = False

    # Initialize Tkinter
    root = tk.Tk()
    root.geometry("500x200")
    root.protocol("WM_DELETE_WINDOW", quit_callback)
    root.title("Normal Modes Control Panel")
    main_dialog = tk.Frame(root)

    # Create and Place Pause Button
    pause_button = tk.Button(main_dialog, text="Pause", command=togglePause)
    pause_button.grid(row=0, column=0)

    # Create Position Inputs and Labels
    position_variables = [tk.DoubleVar() for i in range(3)]
    position_entries = [tk.Entry(main_dialog, width=6,
                                 textvariable=position_variables[i])
                        for i in range(3)]
    position_labels = [tk.Label(main_dialog, text=f"x_{i}")
                       for i in range(3)]

    # Create Velocity Inputs and Labels
    velocity_variables = [tk.DoubleVar() for i in range(3)]
    velocity_entries = [tk.Entry(main_dialog, width=6,
                                 textvariable=velocity_variables[i])
                        for i in range(3)]
    velocity_labels = [tk.Label(main_dialog, text=f"v_{i}")
                       for i in range(3)]

    # Create Mode Amplitude Inputs and Labels
    amplitude_variables = [tk.DoubleVar() for i in range(3)]
    amplitude_labels = [tk.Label(main_dialog, text=f"Mode {i}: Amplitude") for i in range(3)]
    amplitude_entries = [tk.Entry(main_dialog, width=6,
                                 textvariable=amplitude_variables[i])
                         for i in range(3)]

    # Create Mode Phase Inputs and Labels
    phase_variables = [tk.DoubleVar() for i in range(3)]
    phase_labels = [tk.Label(main_dialog, text=f"Mode {i}: Phase") for i in range(3)]
    phase_entries = [tk.Entry(main_dialog, width=6,
                                  textvariable=phase_variables[i])
                         for i in range(3)]

    for i in range(3):
        start_row = 1
        # Place Position Elements
        position_labels[i].grid(row=start_row + i, column=0)
        position_entries[i].grid(row=start_row + i, column=1)

        # Place Velocity Elements
        velocity_labels[i].grid(row=start_row + i, column=2)
        velocity_entries[i].grid(row=start_row + i, column=3)

        amplitude_labels[i].grid(row=start_row + i, column=4)
        amplitude_entries[i].grid(row=start_row + i, column=5)

        phase_labels[i].grid(row=start_row + i, column=6)
        phase_entries[i].grid(row=start_row + i, column=7)

        position_variables[i].set(initial_coords[i])
        velocity_variables[i].set(initial_speeds[i])

        amplitude = np.linalg.norm(coefficients[i])
        phase = np.rad2deg(np.arctan2(coefficients[i][1], coefficients[i][0]))

        amplitude_variables[i].set(round(amplitude * 100, 3))
        phase_variables[i].set(phase)


    main_dialog.pack(fill=tk.BOTH, expand=True)
    ended = False

    while not ended:
        # Tkinter Events
        try:
            main_dialog.update()
        except:
            print("Dialog Error")

        # PyGame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ended = True

        if not paused:
            t += TIME_STEP
        else:
            try:

                initial_coords = [position_variables[i].get() for i in range(3)]
                initial_speeds = [velocity_variables[i].get() for i in range(3)]
                coefficients = calculate_coefficients(omegas, modes,
                                                      equilib_coords,
                                                      initial_coords,
                                                      initial_speeds)
                t = 0

            except:
                pass

        # Calculate New Positions
        deltas = calculate_positions(omegas, modes, coefficients, t)
        velocities = calculate_velocities(omegas, modes, coefficients, t)

        coords = [((equilib_coords[0] + deltas[0]) * WIDTH, HEIGHT // 2),
                  ((equilib_coords[1] + deltas[1]) * WIDTH, HEIGHT // 2),
                  ((equilib_coords[2] + deltas[2]) * WIDTH, HEIGHT // 2)]

        if not paused:
            for i in range(3):
                position_variables[i].set(round(coords[i][0] / WIDTH, 3))
                velocity_variables[i].set(round(velocities[i], 3))

        # PyGame Display Modes
        screen.fill((255, 255, 255))

        pygame.draw.line(screen, SPRING_COLOR, coords[0], coords[1])
        pygame.draw.line(screen, SPRING_COLOR, coords[1], coords[2])

        pygame.draw.circle(screen, MASS_COLOR, (coords[0][0], coords[0][1]), 20)
        pygame.draw.circle(screen, MASS_COLOR, (coords[1][0], coords[1][1]), 20)
        pygame.draw.circle(screen, MASS_COLOR, (coords[2][0], coords[2][1]), 20)

        pygame.display.flip()

    pygame.quit()
    main_dialog.destroy()
