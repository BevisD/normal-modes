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
        self.initial_speeds = [0.0 for _ in range(self.n)]
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
        coeffs = np.array([(0, 0) for _ in self.omegas], dtype=np.float64)
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


class Pygame:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode([WIDTH, HEIGHT])
        self.paused = False

    def display_masses(self, coordinates):
        # PyGame Display Modes
        self.screen.fill((255, 255, 255))

        for i in range(N_MASSES - 1):
            pygame.draw.line(self.screen, SPRING_COLOR,
                             coordinates[i], coordinates[i + 1])

        for coord in coords:
            pygame.draw.circle(self.screen, MASS_COLOR, coord, 20)

        pygame.display.flip()


class GUI:
    def __init__(self):
        # Initialize Tkinter
        self.root = tk.Tk()
        self.root.geometry(f"425x{(N_MASSES + 1) * 22}")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_callback)
        self.root.title("Normal Modes Control Panel")
        self.main_dialog = tk.Frame(self.root)

        self.update_in_progress = False

        # Create Position Inputs and Labels
        self.position_variables = [tk.DoubleVar() for _ in range(N_MASSES)]
        self.position_entries = [tk.Entry(self.main_dialog, width=6,
                                          textvariable=self.position_variables[i])
                                 for i in range(N_MASSES)]
        self.position_labels = [tk.Label(self.main_dialog, text=f"x_{i}")
                                for i in range(N_MASSES)]

        # Create Velocity Inputs and Labels
        self.velocity_variables = [tk.DoubleVar() for _ in range(N_MASSES)]
        self.velocity_entries = [tk.Entry(self.main_dialog, width=6,
                                          textvariable=self.velocity_variables[i])
                                 for i in range(N_MASSES)]
        self.velocity_labels = [tk.Label(self.main_dialog, text=f"v_{i}")
                                for i in range(N_MASSES)]

        # Create Mode Amplitude Inputs and Labels
        self.amplitude_variables = [tk.DoubleVar() for _ in range(N_MASSES)]
        self.amplitude_labels = [tk.Label(self.main_dialog, text=f"Mode {i}: Amplitude")
                                 for i in range(N_MASSES)]
        self.amplitude_entries = [tk.Entry(self.main_dialog, width=6,
                                           textvariable=self.amplitude_variables[i])
                                  for i in range(N_MASSES)]

        # Create Mode Phase Inputs and Labels
        self.phase_variables = [tk.DoubleVar() for _ in range(N_MASSES)]
        self.phase_labels = [tk.Label(self.main_dialog, text=f"Mode {i}: Phase")
                             for i in range(N_MASSES)]
        self.phase_entries = [tk.Entry(self.main_dialog, width=6,
                                       textvariable=self.phase_variables[i])
                              for i in range(N_MASSES)]

        # Create and Place Pause Button
        self.pause_button = tk.Button(self.root, text="Pause", command=toggle_pause)

        phases = modes.coefficients_to_phases()
        self.display_mode_phases(phases)

        for i in range(N_MASSES):
            start_row = 1
            # Place Position Elements
            self.position_labels[i].grid(row=start_row + i, column=0)
            self.position_entries[i].grid(row=start_row + i, column=1)

            # Place Velocity Elements
            self.velocity_labels[i].grid(row=start_row + i, column=2)
            self.velocity_entries[i].grid(row=start_row + i, column=3)

            self.amplitude_labels[i].grid(row=start_row + i, column=4)
            self.amplitude_entries[i].grid(row=start_row + i, column=5)

            self.phase_labels[i].grid(row=start_row + i, column=6)
            self.phase_entries[i].grid(row=start_row + i, column=7)

            self.position_variables[i].set(modes.initial_coords[i])
            self.velocity_variables[i].set(modes.initial_speeds[i])

            self.position_variables[i].trace_add("write", self.set_position_function(i))
            self.velocity_variables[i].trace_add("write", self.set_velocity_function(i))
            self.amplitude_variables[i].trace_add("write", self.set_mode_function(i))
            self.phase_variables[i].trace_add("write", self.set_mode_function(i))

            self.phase_entries[i].config(state=tk.DISABLED)
            self.amplitude_entries[i].config(state=tk.DISABLED)

        self.main_dialog.pack(fill=tk.BOTH, expand=True)
        self.pause_button.pack(side=tk.LEFT)
        self.ended = False

    def quit_callback(self):
        self.ended = True

    def set_position_function(self, id):
        def set_position(name, index, mode):
            if self.update_in_progress or not game.paused:
                return

            if modes.t != 0:
                modes.initial_coords = modes.current_coords

            try:
                modes.initial_coords[id] = self.position_variables[id].get()
            except:
                pass

            modes.initial_speeds = modes.current_speeds

            modes.t = 0
            modes.coefficients = modes.calculate_coefficients()

            phases = modes.coefficients_to_phases()

            self.update_in_progress = True
            self.display_mode_phases(phases)
            self.update_in_progress = False

        return set_position

    def set_velocity_function(self, id):
        def set_velocity(name, index, mode):
            if self.update_in_progress or not game.paused:
                return

            if modes.t != 0:
                modes.initial_coords = modes.current_coords
                modes.initial_speeds = modes.current_speeds

            try:
                modes.initial_speeds[id] = self.velocity_variables[id].get()
            except:
                pass

            modes.t = 0
            modes.coefficients = modes.calculate_coefficients()

            phases = modes.coefficients_to_phases()

            self.update_in_progress = True
            self.display_mode_phases(phases)
            self.update_in_progress = False

        return set_velocity

    def set_mode_function(self, id):
        def set_mode(name, index, mode):
            if self.update_in_progress or not game.paused:
                return

            try:
                modes.t = 0
                phase = np.deg2rad(self.phase_variables[id].get())
                amplitude = self.amplitude_variables[id].get() / 100

                modes.coefficients[id][0] = amplitude * np.cos(phase)
                modes.coefficients[id][1] = amplitude * np.sin(phase)

                deltas = modes.calculate_positions()
                speeds = modes.calculate_velocities()

                self.update_in_progress = True
                for i in range(N_MASSES):
                    self.position_variables[i].set(round(deltas[i], 3))
                    self.velocity_variables[i].set(round(speeds[i], 3))
                self.update_in_progress = False
            except:
                pass

        return set_mode

    def display_mode_phases(self, phases):
        for i in range(N_MASSES):
            self.phase_variables[i].set(phases[i][0])
            self.amplitude_variables[i].set(round(100 * phases[i][1], 3))


def toggle_pause():
    if game.paused:
        state = tk.DISABLED
        message = "Pause"
    else:
        state = tk.NORMAL
        message = "Play"

    for i in range(N_MASSES):
        gui.amplitude_entries[i].config(state=state)
        gui.phase_entries[i].config(state=state)

    gui.pause_button.config(text=message)
    game.paused = not game.paused


if __name__ == "__main__":
    modes = NormalModeSystem(N_MASSES)
    game = Pygame()
    gui = GUI()

    while not gui.ended:
        # Tkinter Events
        try:
            gui.main_dialog.update()
        except:
            print("Dialog Error")

        # PyGame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ended = True

        if not game.paused:
            modes.t += TIME_STEP

        # Calculate New Positions
        positions = modes.calculate_positions()
        velocities = modes.calculate_velocities()

        coords = [(positions[i] * WIDTH, HEIGHT // 2)
                  for i in range(N_MASSES)]

        game.display_masses(coords)

        if not game.paused:
            for i in range(N_MASSES):
                gui.position_variables[i].set(round(coords[i][0] / WIDTH, 3))
                gui.velocity_variables[i].set(round(velocities[i], 3))

    pygame.quit()
    gui.main_dialog.destroy()
