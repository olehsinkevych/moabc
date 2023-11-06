import numpy as np


def _odd(start: int, end: int) -> list[int]:
    nums = []
    for num in range(start, end + 1):
        if num % 2 != 0:
            nums.append(num)
    return nums


def _even(start: int, end: int) -> list[int]:
    nums = []
    for num in range(start, end + 1):
        if num % 2 == 0:
            nums.append(num)
    return nums


def uf_1(x: np.array, dimension: int) -> list[float]:
    j1 = np.array(_odd(start=1, end=dimension-1))
    j2 = np.array(_even(start=1, end=dimension-1))

    def f1():
        if len(j1) > 0:
            return x[0] + (2 / len(j1)) * np.sum((x[j1] - np.sin(6 * np.pi * x[0] + j1*np.pi/dimension))**2)
        else:
            return x[0]

    def f2():
        if len(j2) > 0:
            return 1 - np.sqrt(x[0]) + (2 / len(j2)) * np.sum((x[j2] - np.sin(6 * np.pi * x[0] + j2 * np.pi / dimension))**2)
        else:
            return 1 - np.sqrt(x[0])
    return [f1(), f2()]
