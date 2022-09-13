import numpy as np
import matplotlib.pyplot as plt
import pygame

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

    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    running = True

    while running:
        # PyGame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Calculate New Positions
        deltas = calculate_positions(omegas, modes, coefficients, t)

        # PyGame Display Modes
        screen.fill((255, 255, 255))

        coords = [((equilib_coords[0] + deltas[0]) * WIDTH, HEIGHT // 2),
                  ((equilib_coords[1] + deltas[1]) * WIDTH, HEIGHT // 2),
                  ((equilib_coords[2] + deltas[2]) * WIDTH, HEIGHT // 2)]

        pygame.draw.line(screen, SPRING_COLOR, coords[0], coords[1])
        pygame.draw.line(screen, SPRING_COLOR, coords[1], coords[2])

        pygame.draw.circle(screen, MASS_COLOR, (coords[0][0], coords[0][1]), 20)
        pygame.draw.circle(screen, MASS_COLOR, (coords[1][0], coords[1][1]), 20)
        pygame.draw.circle(screen, MASS_COLOR, (coords[2][0], coords[2][1]), 20)

        pygame.display.flip()
        t += TIME_STEP

    pygame.quit()
