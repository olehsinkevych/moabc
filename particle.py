"""Implements particle as a solution of multi-objective problem"""

from typing import TypeVar, Callable

import numpy as np

TParticle = TypeVar("TParticle", bound="Particle")

class Particle:
    """Particle as a solution to multi-objective optimization"""

    # slots are used to optimize class performance
    __slots__ = ("id", "dim", "position",
                 "cost", "dominated", "l_bound", "u_bound")

    def __init__(self, id: int, dim: int,
                 l_bound: np.array = np.array([]),
                 u_bound: np.array = np.array([]),
                 dominated: bool = False) -> None:
        self.id = id  # particle id
        self.dim = dim  # dimension of the problem
        self.position = np.zeros(dim)  # initial solution x
        self.cost = None  # objective functions values
        self.dominated = dominated  # domination label
        self.l_bound = l_bound  # lower bound for each x component
        self.u_bound = u_bound  # upper bound for each x component

    def initialize_position(self) -> np.array:
        """Initialize first random position."""
        for j in range(self.dim):
            lb = self.l_bound[j]
            ub = self.u_bound[j]
            self.position[j] = lb + np.random.uniform(low=0, high=1)*(ub - lb)

    def set_cost(self, f: Callable) -> None:
        """Calculates cost/objectives"""
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
        return (f"Particle(id={self.id}, dim={self.sim} \n "
                f"position={self.position}, \n cost={self.cost}, "
                f"dominated={self.dominated} \n "
                f"l_bound={self.l_bound}, \n"
                f"u_bound={self.u_bound}")
