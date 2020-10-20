import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, point, box):
        self.dt=dt

        # Vecteur d'etat initial
        self.E=np.matrix([[point[0]], [point[1]], [0], [0], [box[0]], [box[1]]])

        # Matrice de transition
        self.A=np.matrix([[1, 0, self.dt, 0, 0, 0],
                          [0, 1, 0, self.dt, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

        # Matrice d'observation, on observe que x et y
        self.H=np.matrix([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

        v=1E-5
        #v=1
        self.Q=np.matrix([[v, 0, 0, 0, 0, 0],
                          [0, v, 0, 0, 0, 0],
                          [0, 0, v, 0, 0, 0],
                          [0, 0, 0, v, 0, 0],
                          [0, 0, 0, 0, v, 0],
                          [0, 0, 0, 0, 0, v]])

        v=1E-5
        #v=1
        self.R=np.matrix([[v, 0, 0, 0],
                          [0, v, 0, 0],
                          [0, 0, v, 0],
                          [0, 0, 0, v]])

        self.P=np.eye(self.A.shape[1])

    def predict(self):
        self.E=np.dot(self.A, self.E)
        # Calcul de la covariance de l'erreur
        self.P=np.dot(np.dot(self.A, self.P), self.A.T)+self.Q
        return self.E

    def update(self, z):
        # Calcul du gain de Kalman
        S=np.dot(self.H, np.dot(self.P, self.H.T))+self.R
        K=np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Correction / innovation
        self.E=np.round(self.E+np.dot(K, (z-np.dot(self.H, self.E))))
        I=np.eye(self.H.shape[1])
        self.P=(I-(K*self.H))*self.P

        return self.E

class KalmanFilter_old(object):
    def __init__(self, dt, point, box):
        self.dt=dt

        # Vecteur d'etat initial
        self.E=np.matrix([[point[0]], [point[1]], [0], [0], [box[0]], [box[1]], [0], [0]])

        # Matrice de transition
        self.A=np.matrix([[1, 0, self.dt, 0, 0, 0, 0, 0],
                          [0, 1, 0, self.dt, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, self.dt, 0],
                          [0, 0, 0, 0, 0, 1, 0, self.dt],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])

        # Matrice d'observation, on observe que x et y
        self.H=np.matrix([[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0]])

        self.Q=np.matrix([[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])

        self.R=np.matrix([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        self.P=np.eye(self.A.shape[1])

    def predict(self):
        self.E=np.dot(self.A, self.E)
        # Calcul de la covariance de l'erreur
        self.P=np.dot(np.dot(self.A, self.P), self.A.T)+self.Q
        return self.E

    def update(self, z):
        # Calcul du gain de Kalman
        S=np.dot(self.H, np.dot(self.P, self.H.T))+self.R
        K=np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Correction / innovation
        self.E=np.round(self.E+np.dot(K, (z-np.dot(self.H, self.E))))
        I=np.eye(self.H.shape[1])
        self.P=(I-(K*self.H))*self.P

        return self.E
