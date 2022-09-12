import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    T = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    V = np.array([[1, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])

    eigenvalues, eigenvectors = np.linalg.eigh(V)
    omegas = np.sqrt(eigenvalues)

    print(omegas)

    coefficients = [(0, 0), (1, 0), (1, 0)]

    t = 0
    times = [0]
    positions = np.array([[0,0,0]])
    while t < 10:
        p = np.sum(
            [eigenvectors[:, i] * coefficients[i][0] * np.sin(omegas[i] * t) +
             eigenvectors[:, i] * coefficients[i][1] * np.cos(omegas[i] * t)
             for i in range(len(coefficients))], axis=0)

        positions = np.vstack((positions, p))
        times.append(t)

        t += 0.1

    plt.plot(times, positions[:, 0] - 1)
    plt.plot(times, positions[:, 1])
    plt.plot(times, positions[:, 2] + 1)

    plt.show()
