import numpy as np
import torch

class KalmanFilter:
    def __init__(self, x0, dt=1.0/30, std_acc=0.1, std_meas=1):
        self.dt = dt
        # measurement matrix
        self.H = torch.zeros(3, 6)
        self.H[:3, :3] = torch.eye(3)

        # state transition matrix
        self.A = torch.eye(6)
        self.A[:3, 3:6] = torch.eye(3)

        # acceleration covariance matrix
        self.Q = torch.zeros(6, 6)
        self.Q[:3, :3] = dt**4/4.0 * torch.eye(3)
        self.Q[:3, 3:6] = dt**3/2.0 * torch.eye(3)
        self.Q[3:6, :3] = dt**3/2.0 * torch.eye(3)
        self.Q[3:6, 3:6] = dt**2 * torch.eye(3)
        self.Q *= std_acc

        # measurement covariance matrix
        self.R = torch.eye(3)*std_meas
        self.P = torch.eye(self.A.shape[1])
        self.x = x0

    def predict(self):
        # Ref :Eq.(9) and Eq.(10)
        # Update time state
        self.x = self.A @ self.x # + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = (self.A @ self.P) @ self.A.T + self.Q

        return self.x[:3]

    def update(self, z):
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        S = self.H @ (self.P @ self.H.T) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = (self.P @ self.H.T) @ torch.tensor(np.linalg.inv(S.cpu().numpy()), dtype=torch.float32)  # Eq.(11)
        self.x = self.x + K @ (z - (self.H @ self.x))  # Eq.(12)
        I = torch.eye(self.H.shape[1])
        #self.P = (I - (K * self.H)) * self.P  # Eq.(13)
        self.P = (I - K @ self.H) @ self.P  # Eq.(13)
