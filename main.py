import numpy as np
import pygame
import tkinter as tk

WIDTH = 600
HEIGHT = 600
MASS_COLOR = (0, 0, 255)
SPRING_COLOR = (0, 0, 0)
TIME_STEP = 0.005
N_MASSES = 6

update_in_progress = False


class NormalModeSystem:
    def __init__(self, n):
        # Coordinates given as fractions of window width
        self.n = n
        self.equilib_coords = [(i + 1) / (self.n + 1) for i in range(self.n)]
        self.initial_coords = [(i + 1) / (self.n + 1) for i in range(self.n)]
        self.initial_speeds = [0.0 for i in range(self.n)]
        self.initial_speeds[0] = 0.05
        self.initial_speeds[-1] = -0.05

        self.current_coords = self.initial_coords
        self.current_speeds = self.initial_speeds

        # Kinetic Energy Tensor
        self.T = np.diag(np.ones(self.n))

        # Potential Energy Tensor
        self.V = np.zeros((self.n, self.n,))
        for i in range(self.n - 1):
            self.V[i:i + 2, i:i + 2] += np.array([[1, -1], [-1, 1]])

        # Calculate Normal Modes of the System
        self.omegas, self.modes = self.calculate_normal_modes()

        # Calculate coefficients of each normal mode
        self.coefficients = self.calculate_coefficients()

        self.t = 0

    def calculate_normal_modes(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.V)
        omegas = np.sqrt(np.abs(eigenvalues))
        return np.round(omegas, 6), np.round(eigenvectors, 6)

    def calculate_coefficients(self):
        coeffs = np.array([(0, 0) for j in self.omegas], dtype=np.float64)
        offsets = np.subtract(self.initial_coords, self.equilib_coords)

        for j in range(self.n):
            coeffs[j][1] = np.round(np.dot(offsets, self.modes[:, j]), 6)
            if self.omegas[j] == 0:
                coeffs[j][0] = np.round(np.dot(self.initial_speeds, self.modes[:, j]), 6)
            else:
                coeffs[j][0] = np.round(
                    np.dot(self.initial_speeds, self.modes[:, j]) / self.omegas[j], 6)

        return coeffs

    def calculate_positions(self):
        offsets = np.zeros(N_MASSES)
        for j in range(self.n):
            if self.omegas[j] == 0:
                offsets += self.modes[:, j] * self.coefficients[j][0] * self.t + \
                           self.modes[:, j] * self.coefficients[j][1]
            else:
                offsets += self.modes[:, j] * self.coefficients[j][0] \
                           * np.sin(self.omegas[j] * self.t) + \
                           self.modes[:, j] * self.coefficients[j][1] \
                           * np.cos(self.omegas[j] * self.t)

        self.current_coords = np.add(offsets, self.equilib_coords)
        return self.current_coords

    def calculate_velocities(self):
        speeds = np.zeros(N_MASSES)
        for j in range(self.n):
            if self.omegas[j] == 0:
                speeds += self.modes[:, j] * self.coefficients[j][0]
            else:
                speeds += self.modes[:, j] * self.coefficients[j][0] * \
                          np.cos(self.omegas[j] * self.t) * self.omegas[j] - \
                          self.modes[:, j] * self.coefficients[j][1] * \
                          np.sin(self.omegas[j] * self.t) * self.omegas[j]
        self.current_speeds = speeds
        return speeds

    def coefficients_to_phases(self):
        phases = []
        for coefficient in self.coefficients:
            phase = np.rad2deg(np.arctan2(coefficient[1], coefficient[0]))
            amplitude = np.linalg.norm(coefficient)
            phases.append([phase, amplitude])
        return phases


def quit_callback():
    global ended
    ended = True


def toggle_pause():
    global paused
    if paused:
        state = tk.DISABLED
        message = "Pause"
    else:
        state = tk.NORMAL
        message = "Play"

    for i in range(N_MASSES):
        amplitude_entries[i].config(state=state)
        phase_entries[i].config(state=state)

    pause_button.config(text=message)
    paused = not paused


def set_position_function(id):
    def set_position(name, index, mode):
        global update_in_progress
        if update_in_progress or not paused:
            return

        if modes.t != 0:
            modes.initial_coords = modes.current_coords

        try:
            modes.initial_coords[id] = position_variables[id].get()
        except:
            pass

        modes.initial_speeds = modes.current_speeds

        modes.t = 0
        modes.coefficients = modes.calculate_coefficients()

        phases = modes.coefficients_to_phases()

        update_in_progress = True
        display_mode_phases(phases)
        update_in_progress = False
    return set_position


def set_velocity_function(id):
    def set_velocity(name, index, mode):
        global update_in_progress
        if update_in_progress or not paused:
            return

        if modes.t != 0:
            modes.initial_coords = modes.current_coords
            modes.initial_speeds = modes.current_speeds

        try:
            modes.initial_speeds[id] = velocity_variables[id].get()
        except:
            pass

        modes.t = 0
        modes.coefficients = modes.calculate_coefficients()

        phases = modes.coefficients_to_phases()

        update_in_progress = True
        display_mode_phases(phases)
        update_in_progress = False
    return set_velocity


def set_mode_function(id):
    def set_mode(name, index, mode):
        global update_in_progress
        if update_in_progress or not paused:
            return

        try:
            modes.t = 0
            phase = np.deg2rad(phase_variables[id].get())
            amplitude = amplitude_variables[id].get() / 100

            modes.coefficients[id][0] = amplitude * np.cos(phase)
            modes.coefficients[id][1] = amplitude * np.sin(phase)

            deltas = modes.calculate_positions()
            speeds = modes.calculate_velocities()

            update_in_progress = True
            for i in range(N_MASSES):
                position_variables[i].set(round(deltas[i], 3))
                velocity_variables[i].set(round(speeds[i], 3))
            update_in_progress = False
        except:
            pass
    return set_mode


def display_mode_phases(phases):
    for i in range(N_MASSES):
        phase_variables[i].set(phases[i][0])
        amplitude_variables[i].set(round(100 * phases[i][1], 3))


if __name__ == "__main__":
    modes = NormalModeSystem(N_MASSES)

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

    # Create Position Inputs and Labels
    position_variables = [tk.DoubleVar() for i in range(N_MASSES)]
    position_entries = [tk.Entry(main_dialog, width=6,
                                 textvariable=position_variables[i])
                        for i in range(N_MASSES)]
    position_labels = [tk.Label(main_dialog, text=f"x_{i}")
                       for i in range(N_MASSES)]

    # Create Velocity Inputs and Labels
    velocity_variables = [tk.DoubleVar() for i in range(N_MASSES)]
    velocity_entries = [tk.Entry(main_dialog, width=6,
                                 textvariable=velocity_variables[i])
                        for i in range(N_MASSES)]
    velocity_labels = [tk.Label(main_dialog, text=f"v_{i}")
                       for i in range(N_MASSES)]

    # Create Mode Amplitude Inputs and Labels
    amplitude_variables = [tk.DoubleVar() for i in range(N_MASSES)]
    amplitude_labels = [tk.Label(main_dialog, text=f"Mode {i}: Amplitude") for i in range(N_MASSES)]
    amplitude_entries = [tk.Entry(main_dialog, width=6,
                                  textvariable=amplitude_variables[i])
                         for i in range(N_MASSES)]

    # Create Mode Phase Inputs and Labels
    phase_variables = [tk.DoubleVar() for i in range(N_MASSES)]
    phase_labels = [tk.Label(main_dialog, text=f"Mode {i}: Phase") for i in range(N_MASSES)]
    phase_entries = [tk.Entry(main_dialog, width=6,
                              textvariable=phase_variables[i])
                     for i in range(N_MASSES)]

    # Create and Place Pause Button
    pause_button = tk.Button(root, text="Pause", command=toggle_pause)

    phases = modes.coefficients_to_phases()
    display_mode_phases(phases)

    for i in range(N_MASSES):
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

        position_variables[i].set(modes.initial_coords[i])
        velocity_variables[i].set(modes.initial_speeds[i])

        position_variables[i].trace_add("write", set_position_function(i))
        velocity_variables[i].trace_add("write", set_velocity_function(i))
        amplitude_variables[i].trace_add("write", set_mode_function(i))
        phase_variables[i].trace_add("write", set_mode_function(i))

        phase_entries[i].config(state=tk.DISABLED)
        amplitude_entries[i].config(state=tk.DISABLED)

    main_dialog.pack(fill=tk.BOTH, expand=True)
    pause_button.pack(side=tk.LEFT)
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
            modes.t += TIME_STEP

        # Calculate New Positions
        positions = modes.calculate_positions()
        velocities = modes.calculate_velocities()

        coords = [(positions[i] * WIDTH, HEIGHT // 2)
                  for i in range(N_MASSES)]

        if not paused:
            for i in range(N_MASSES):
                position_variables[i].set(round(coords[i][0] / WIDTH, 3))
                velocity_variables[i].set(round(velocities[i], 3))

        # PyGame Display Modes
        screen.fill((255, 255, 255))

        for i in range(N_MASSES - 1):
            pygame.draw.line(screen, SPRING_COLOR, coords[i], coords[i + 1])

        for coord in coords:
            pygame.draw.circle(screen, MASS_COLOR, coord, 20)

        pygame.display.flip()

    pygame.quit()
    main_dialog.destroy()
