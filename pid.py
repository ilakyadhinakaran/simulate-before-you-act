class PID:
    def __init__(self, kp=0.5, ki=0.05, pmax=1.0):
        self.kp, self.ki, self.pmax, self.i = kp, ki, pmax, 0.0
    def update(self, setpoint, measured):
        e = setpoint - measured
        self.i += e
        u = self.kp * e + self.ki * self.i
        if u < 0 or u > self.pmax: self.i -= e   # anti-windup
        return min(max(u, 0.0), self.pmax)