import numpy as np

class CareerEnv:
    def __init__(self):
        self.max_years = 10
        self.reset()

    def reset(self):
        self.year = 1
        self.skill = 0.3
        self.marks = 85
        self.budget = 1200000
        self.path = 1
        self.interest = 0.8
        self.risk_tolerance = 0.6
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.year,
            self.skill,
            self.marks / 100,
            self.budget / 2000000,
            self.path,
            self.interest,
            self.risk_tolerance
        ], dtype=np.float32)

    def calculate_cds(self):
        interest_match = self.interest
        academic_fit = self.marks / 100
        budget_fit = min(self.budget / 2000000, 1)
        demand_score = 0.85
        risk_alignment = 1 - abs(self.risk_tolerance - (self.path / 2))

        return (
            0.25 * interest_match +
            0.20 * academic_fit +
            0.20 * budget_fit +
            0.15 * demand_score +
            0.10 * risk_alignment +
            0.10 * self.skill
        )

    def check_feasibility(self):
        if self.path == 0:
            return self.marks >= 90 and self.budget >= 3000000
        elif self.path == 1:
            return self.marks >= 70
        return True

    def step(self, action):
        reward = 0

        if action == 0:
            self.skill += 0.1
            reward += 2
        elif action == 1:
            self.path = 0
            reward -= 2
        elif action == 2:
            self.path = 1
            reward += 1
        elif action == 3:
            self.path = 2
            reward += 2

        cds = self.calculate_cds()
        reward += cds * 10

        if not self.check_feasibility():
            reward -= 10

        if self.path == 0:
            reward += self.skill * 8
            if self.skill < 0.5:
                reward -= 6
        elif self.path == 2:
            reward += 3

        self.year += 1
        self.skill = min(self.skill, 1.0)

        if self.year > self.max_years:
            self.done = True
            if self.path == 0:
                reward += int(self.skill * 60)
            elif self.path == 1:
                reward += int(self.skill * 40)
            else:
                reward += int(self.skill * 25)

        return self._get_state(), reward, self.done, {}
