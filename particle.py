"""Implements particle as a solution of multi-objective problem"""

from typing import TypeVar, Callable

import numpy as np

TParticle = TypeVar("TParticle", bound="Particle")


class Particle:
    """Particle as a solution to multi-objective optimization"""

    __slots__ = ("id", "dim", "velocity", "position",
                 "cost", "dominated", "is_improved",
                 "trial", "l_bound", "u_bound")

    def __init__(self, id: int, dim: int,
                 l_bound: np.array = np.array([]),
                 u_bound: np.array = np.array([]),
                 dominated: bool = False,
                 velocity: float = 0) -> None:
        self.id = id
        self.dim = dim
        self.velocity = velocity
        self.position = np.zeros(dim)
        self.cost = None
        self.dominated = dominated
        self.is_improved = False
        self.trial = 0
        self.l_bound = l_bound
        self.u_bound = u_bound

    def initialize_position(self) -> np.array:
        """Initialize first random position."""
        for j in range(self.dim):
            lb = self.l_bound[j]
            ub = self.u_bound[j]
            self.position[j] = lb + np.random.uniform(low=0, high=1) * (ub - lb)

    def set_cost(self, f: Callable) -> None:
        """Calculates cost"""
        self.cost = f(x=self.position, dimension=self.dim)

    def dominates(self, other: TParticle) -> bool:
        """Checks if particle dominated other"""
        all_ = [True if self.cost[i] <= other.cost[i] else
                False for i in range(len(self.cost))]
        any_ = [True if self.cost[i] < other.cost[i] else
                False for i in range(len(self.cost))]
        if all(all_) and any(any_):
            return True
        else:
            return False

    @staticmethod
    def check_domination(particles: list[TParticle]) -> None:
        """Perform Pareto domination check for a list of Particles"""
        num = len(particles)
        for i in range(num):
            particles[i].dominated = False
            for j in range(0, i):
                if not particles[j].dominated:
                    if particles[i].dominates(particles[j]):
                        particles[j].dominated = True
                    elif particles[j].dominates(particles[i]):
                        particles[i].dominated = True
                        break

    def __repr__(self) -> str:
        """Representation of the Particle (self)"""
        return (f"Particle(velocity={self.velocity}, \n "
                f"position={self.position}, \n cost={self.cost}, "
                f"dominated={self.dominated}, \n trial={self.trial})")
