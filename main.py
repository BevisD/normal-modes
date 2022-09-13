import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

WIDTH = 600
HEIGHT = 600
CONTROL_WIDTH = 200

MASS_COLOR = (0, 0, 255)
SPRING_COLOR = (0, 0, 0)
TIME_STEP = 0.1


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



class Mass:
    def __init__(self, canvas, x, y, r, fill="#000000", outline="#000000"):
        self.canvas = canvas
        self.fill = fill
        self.outline = outline

        self.id = self.canvas.create_oval(x - r, y - r,
                                       x + r, y + r,
                                       fill=fill, outline=outline)

        self.r = r
        self._x = x
        self._y = y
        self.x = x
        self.y = y


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self.canvas.moveto(self.id, value - self.r, self.y - self.r)
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self.canvas.moveto(self.id, self.x - self.r, value - self.r)
        self._y = value


class App:
    def __init__(self, main, omegas, modes, coefficients,
                 equilib_coords, inital_coords, time=0):
        self.main = main
        self.omegas = omegas
        self.modes = modes
        self.coefficients = coefficients
        self.equilib_coords = equilib_coords
        self.time = time
        self.running = True

        self.canvas = tk.Canvas(self.main, width=WIDTH, height=HEIGHT,
                                bg="white", bd=1, highlightthickness=0,
                                relief="solid")
        self.canvas.pack(side="left")

        self.masses = [Mass(self.canvas, x * WIDTH, HEIGHT/2, 10) for x in inital_coords]
        self.canvas.pack()
        self.main.after(0, self.animation)

    def animation(self):
        deltas = calculate_positions(self.omegas, self.modes,
                                     self.coefficients, self.time)
        coords = [((self.equilib_coords[0] + deltas[0]) * WIDTH, HEIGHT // 2),
                  ((self.equilib_coords[1] + deltas[1]) * WIDTH, HEIGHT // 2),
                  ((self.equilib_coords[2] + deltas[2]) * WIDTH, HEIGHT // 2)]

        for i in range(len(self.masses)):
            self.masses[i].x = coords[i][0]

        self.time += TIME_STEP

        if self.running:
            self.main.after(10, self.animation)


def pause_animation():
    if app.running:
        app.running = False

    else:
        app.running = True
        app.animation()


if __name__ == "__main__":
    # Coordinates given as fractions of window width
    equilib_coords = [0.25, 0.5, 0.75]
    initial_coords = [0.15, 0.5, 0.85]
    initial_speeds = [0.0, 0.0, 0.0]

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

    window = tk.Tk()
    window.geometry(f"{WIDTH + CONTROL_WIDTH}x{HEIGHT}")

    app = App(window, omegas, modes, coefficients,
              equilib_coords, initial_coords)

    pause = tk.Button(window, text="Pause", command=pause_animation)
    pause.pack()

    window.mainloop()

    # running = True
    #
    # while running:
    #     # PyGame Events
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     # Calculate New Positions
    #
    #
    #     # PyGame Display Modes
    #     screen.fill((255, 255, 255))
    #
    #     coords = [((equilib_coords[0] + deltas[0]) * WIDTH, HEIGHT // 2),
    #               ((equilib_coords[1] + deltas[1]) * WIDTH, HEIGHT // 2),
    #               ((equilib_coords[2] + deltas[2]) * WIDTH, HEIGHT // 2)]
    #
    #     pygame.draw.line(screen, SPRING_COLOR, coords[0], coords[1])
    #     pygame.draw.line(screen, SPRING_COLOR, coords[1], coords[2])
    #
    #     pygame.draw.circle(screen, MASS_COLOR, (coords[0][0], coords[0][1]), 20)
    #     pygame.draw.circle(screen, MASS_COLOR, (coords[1][0], coords[1][1]), 20)
    #     pygame.draw.circle(screen, MASS_COLOR, (coords[2][0], coords[2][1]), 20)
    #
    #     pygame.display.flip()
    #     t += TIME_STEP
    #
    # pygame.quit()
