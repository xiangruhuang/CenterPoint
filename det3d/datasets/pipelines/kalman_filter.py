import numpy as np
import torch

class KalmanFilter:
    def __init__(self, x0, dt=1.0/30, std_acc=0.1, std_meas=1, dtype=torch.float64):
        self.dt = dt
        # measurement matrix
        self.H = torch.zeros(3, 6).to(dtype)
        self.H[:3, :3] = torch.eye(3).to(dtype)

        # state transition matrix
        self.A = torch.eye(6).to(dtype)
        self.A[:3, 3:6] = torch.eye(3).to(dtype)

        # acceleration covariance matrix
        self.Q = torch.zeros(6, 6).to(dtype)
        self.Q[:3, :3] = dt**4/4.0 * torch.eye(3).to(dtype)
        self.Q[:3, 3:6] = dt**3/2.0 * torch.eye(3).to(dtype)
        self.Q[3:6, :3] = dt**3/2.0 * torch.eye(3).to(dtype)
        self.Q[3:6, 3:6] = dt**2 * torch.eye(3).to(dtype)
        self.Q *= std_acc

        # measurement covariance matrix
        self.R = torch.eye(3).to(dtype)*std_meas
        self.P = torch.eye(self.A.shape[1]).to(dtype)
        self.x = x0
        self.dtype = dtype

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
        K = (self.P @ self.H.T) @ torch.tensor(np.linalg.inv(S.cpu().numpy()), dtype=self.dtype)  # Eq.(11)
        self.x = self.x + K @ (z - (self.H @ self.x))  # Eq.(12)
        I = torch.eye(self.H.shape[1]).to(self.dtype)
        #self.P = (I - (K * self.H)) * self.P  # Eq.(13)
        self.P = (I - K @ self.H) @ self.P  # Eq.(13)

class StatelessKalmanFilter:
    def __init__(self, dt=1.0/30, std_acc=0.1, std_meas=1, inplace=True,
                 dtype=torch.float64, device='cuda'):
        self.dt = dt
        # measurement matrix
        self.H = torch.zeros(3, 6, dtype=dtype, device=device)
        self.H[:3, :3] = torch.eye(3, dtype=dtype, device=device)

        # state transition matrix
        self.A = torch.eye(6, dtype=dtype, device=device)
        self.A[:3, 3:6] = torch.eye(3, dtype=dtype, device=device)

        # acceleration covariance matrix
        self.Q = torch.zeros(6, 6, dtype=dtype, device=device)
        self.Q[:3, :3] = dt**4/4.0 * torch.eye(3, dtype=dtype, device=device)
        self.Q[:3, 3:6] = dt**3/2.0 * torch.eye(3, dtype=dtype, device=device)
        self.Q[3:6, :3] = dt**3/2.0 * torch.eye(3, dtype=dtype, device=device)
        self.Q[3:6, 3:6] = dt**2 * torch.eye(3, dtype=dtype, device=device)
        self.Q *= std_acc

        # measurement covariance matrix
        self.R = torch.eye(3, dtype=dtype, device=device)*std_meas
        #self.P = torch.eye(self.A.shape[1]).to(dtype)
        self.dtype = dtype
        self.device = device
        self.inplace = inplace

    def predict(self, x, P):
        """ Predict next state.

        Args:
            x: torch.tensor([N, 6])
            P: torch.tensor([N, 6, 6])
            
        Returns:
            x_pred: torch.tensor([N, 3])
            next_x: torch.tensor([N, 6])
            next_P: torch.tensor([N, 6, 6])
        """
        # Ref :Eq.(9) and Eq.(10)
        if self.inplace:
            # Update time state
            x[:] = x @ self.A.T
            # Calculate error covariance
            # P= A*P*A' + Q
            P[:] = (self.A @ P) @ self.A.T + self.Q
            
            return x[:, :3], x, P
        else:
            # Update time state
            next_x = x @ self.A.T
            # Calculate error covariance
            # P= A*P*A' + Q
            next_P = (self.A @ P) @ self.A.T + self.Q
            
            return next_x[:, :3], next_x, next_P

    def update(self, z, x, P):
        """ Update Kalman Filter state.

        Args:
            z: torch.tensor([N, 3])
            x: torch.tensor([N, 6])
            P: torch.tensor([N, 6, 6])

        Returns:
            next_x: torch.tensor([N, 6])
            next_P: torch.tensor([N, 6, 6])
            
        """
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        S = self.H @ (P @ self.H.T) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = (P @ self.H.T) @ torch.tensor(np.linalg.inv(S.cpu().numpy()),
                                          dtype=self.dtype)  # Eq.(11)
        I = torch.eye(self.H.shape[1], dtype=self.dtype, device=self.device)
        if self.inplace:
            x += (K @ (z-x @ self.H.T).unsqueeze(-1)).squeeze(-1) # Eq.(12)
            P[:] = (I - K @ self.H) @ P  # Eq.(13)
            return x, P
        else:
            next_x += (K @ (z-x @ self.H.T).unsqueeze(-1)).squeeze(-1) # Eq.(12)
            next_P = (I - K @ self.H) @ P  # Eq.(13)
            return next_x, next_P

if __name__ == '__main__':
    def test_kalman_filter(kf, real_track, t):
        predictions = []
        measurements = []
        for x in real_track:
            # Mesurement
            z = x + torch.randn(3) * 5
            measurements.append(z)
            predictions.append(kf.predict())
            kf.update(z)
        predictions = torch.stack(predictions, dim=0)
        measurements = torch.stack(measurements, dim=0)
        fig = plt.figure()
        fig.suptitle('Example of Kalman filter for tracking a moving object in 1-D', fontsize=20)
        plt.plot(t[:, 0], measurements[:, 0], label='Measurements', color='b',linewidth=0.5)
        plt.plot(t[:, 0], np.array(real_track[:, 0]), label='Real Track', color='y', linewidth=1.5)
        plt.plot(t[:, 0], np.squeeze(predictions[:, 0]), label='Kalman Filter Prediction', color='r', linewidth=1.5)
        plt.xlabel('Time (s)', fontsize=20)
        plt.ylabel('Position (m)', fontsize=20)
        plt.legend()
        plt.show()

    def test_stateless_kalman_filter(kf, real_track, t):
        predictions = []
        measurements = []

        n = 40
        x = torch.zeros(n, 6, dtype=torch.float64)
        P = torch.eye(6, dtype=torch.float64).repeat(n, 1, 1)
        for r in real_track:
            # Mesurement
            z = r + torch.randn(n, 3) * 5
            measurements.append(z)
            x_pred, x, P = kf.predict(x, P)
            predictions.append(x_pred.clone())
            x, P = kf.update(z, x, P)

        predictions = torch.stack(predictions, dim=0)[:, 2]
        measurements = torch.stack(measurements, dim=0)[:, 2]
        fig = plt.figure()
        fig.suptitle('Example of Kalman filter for tracking a moving object in 1-D', fontsize=20)
        plt.plot(t[:, 0], measurements[:, 0], label='Measurements', color='b',linewidth=0.5)
        plt.plot(t[:, 0], np.array(real_track[:, 0]), label='Real Track', color='y', linewidth=1.5)
        plt.plot(t[:, 0], np.squeeze(predictions[:, 0]), label='Kalman Filter Prediction', color='r', linewidth=1.5)
        plt.xlabel('Time (s)', fontsize=20)
        plt.ylabel('Position (m)', fontsize=20)
        plt.legend()
        plt.show()

    import matplotlib.pyplot as plt
    dt = 0.1
    t = torch.arange(0, 100, dt).double().repeat(3, 1).T
    # Define a model track
    real_track = 0.1*((t**2) - t)
    u = 0
    std_acc = 0.25     # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
    std_meas = 5.0    # and standard deviation of the measurement is 1.2 (m)
    # create KalmanFilter object
    x0 = torch.zeros(6, dtype=torch.float64)
    kf = KalmanFilter(x0, dt, std_acc, std_meas)
    #test_kalman_filter(kf, real_track, t)

    # test Stateless Kalman Filter
    skf = StatelessKalmanFilter(dt, std_acc, std_meas, device='cpu')
    test_stateless_kalman_filter(skf, real_track, t)
