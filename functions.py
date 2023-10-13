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
    f1 = x[0] + (2 / len(j1)) * np.sum(x[j1] - np.sin(6 * np.pi * x[0] + j1*np.pi/dimension))**2
    f2 = 1 - np.sqrt(x[0]) + (2 / len(j2)) * np.sum(x[j2] - np.sin(6 * np.pi * x[0] + j2 * np.pi / dimension)) ** 2
    return [f1, f2]
