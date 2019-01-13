import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class PLMS_solver:
    def __init__(self,X, y, noise_var, diffusion_var):
        self.X = X
        self.y = y
        self.noise_var = noise_var
        self.diffusion_var = diffusion_var
        self.mu = np.ones_like(self.X) * 0.1
        self.sigma_sq = np.zeros(self.X.shape[0])

    def Learn(self):
        for i in range(1,self.X.shape[0]):
            #Mean Update
            sigma_sum = self.sigma_sq[i-1] + self.diffusion_var
            x_norm_sq = np.linalg.norm(self.X[i,:])**2
            #eta: Adaptive Learning Rate
            eta = sigma_sum/(sigma_sum * x_norm_sq + self.noise_var)
            prev_error = self.y[i] - np.dot(self.X[i,:], self.mu[i-1,:])
            self.mu[i,:] = self.mu[i-1,:] + eta * prev_error * self.X[i,:]
            #Var Update
            self.sigma_sq[i] = (1 - (eta * x_norm_sq)/X.shape[1]) * sigma_sum

    def TrackOutput(self):
        pred = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            pred[i] = np.dot(self.X[i,:], self.mu[i,:])
        return pred

    def StationaryError(self, w_opt):
        err = np.zeros(self.X.shape[0])

        opt = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            opt[i] = np.dot(self.X[i,:], w_opt)

        for i in range(self.X.shape[0]):
            pred = np.zeros(self.X.shape[0])
            for j in range(self.X.shape[0]):
                pred[j] = np.dot(self.X[j,:], self.mu[i,:])
            err[i] = np.linalg.norm(pred - opt)**2
        return err

class RLS_solver:
    def __init__(self,X, y, noise_var, diffusion_var):
        self.X = X
        self.y = y
        self.noise_var = noise_var
        self.diffusion_var = diffusion_var
        self.sigma = np.matrix(np.identity(X.shape[1]))
        self.K = np.matrix(np.identity(X.shape[1]))
        self.mu = np.ones_like(self.X) * 0.1

    def Learn(self):
        for i in range(1,self.X.shape[0]):
            #Mean Update
            sigma_sum = self.sigma + self.diffusion_var * np.matrix(np.identity(X.shape[1]))
            x_i = np.matrix(X[i,:]).T
            #K: Adaptive Learning Rate
            self.K = sigma_sum/(x_i.T * sigma_sum * x_i + self.noise_var)
            self.sigma = (np.matrix(np.identity(X.shape[1]) - self.K*x_i*x_i.T))*sigma_sum
            prev_error = self.y[i] - np.dot(self.X[i,:], self.mu[i-1,:])
            self.mu[i,:] = self.mu[i-1,:] + prev_error * np.array(self.K * x_i).flatten()

    def TrackOutput(self):
        pred = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            pred[i] = np.dot(self.X[i,:], self.mu[i,:])
        return pred

    def StationaryError(self, w_opt):
        err = np.zeros(self.X.shape[0])

        opt = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            opt[i] = np.dot(self.X[i,:], w_opt)

        for i in range(self.X.shape[0]):
            pred = np.zeros(self.X.shape[0])
            for j in range(self.X.shape[0]):
                pred[j] = np.dot(self.X[j,:], self.mu[i,:])
            err[i] = np.linalg.norm(pred - opt)**2
        return err

class LMS_solver:
    def __init__(self,X, y, noise_var, diffusion_var):
        self.X = X
        self.y = y
        self.noise_var = noise_var
        self.diffusion_var = diffusion_var
        self.mu = np.ones_like(self.X) * 0.1

    def Learn(self):
        #Initialize learning rate
        Rxx = np.cov(self.X, rowvar=False, bias=True)
        w, v = eigh(Rxx)
        eta = 0.02/w[-1]
        for i in range(1,self.X.shape[0]):
            #Mean Update
            x_norm_sq = np.linalg.norm(self.X[i,:])**2
            prev_error = self.y[i] - np.dot(self.X[i,:], self.mu[i-1,:])
            self.mu[i,:] = self.mu[i-1,:] + eta * prev_error * self.X[i,:]

    def TrackOutput(self):
        pred = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            pred[i] = np.dot(self.X[i,:], self.mu[i,:])
        return pred

    def StationaryError(self, w_opt):
        err = np.zeros(self.X.shape[0])

        opt = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            opt[i] = np.dot(self.X[i,:], w_opt)

        for i in range(self.X.shape[0]):
            pred = np.zeros(self.X.shape[0])
            for j in range(self.X.shape[0]):
                pred[j] = np.dot(self.X[j,:], self.mu[i,:])
            err[i] = np.linalg.norm(pred - opt)**2
        return err



#Create Stationary Input
M = 200
C = 8
sigma_noise = 0.1
sigma_diffuse = 0.05
X = np.array([[1,2,3,4,5] for i in range(M)]).reshape((-1,C))


#Output w/o noise
w_opt = np.random.normal(0, 1, X.shape[1])
y = np.zeros(X.shape[0])
for i in range(0, X.shape[0]):
    y[i] = np.dot(X[i,:], w_opt)
#Adding noise
v = np.random.normal(0, sigma_noise, len(y))

#Training PLMS Filter
solver = PLMS_solver(X,y + v,sigma_noise, sigma_diffuse)
solver.Learn()
error_plms = solver.StationaryError(w_opt)

#Training RLS Filter
solver = RLS_solver(X,y + v,sigma_noise, sigma_diffuse)
solver.Learn()
error_rls = solver.StationaryError(w_opt)

#Training LMS Filter
solver = LMS_solver(X,y + v,sigma_noise, sigma_diffuse)
solver.Learn()
error_lms = solver.StationaryError(w_opt)

its = np.arange(X.shape[0])
plt.title('Stationary, noise = 0.1,diffusion = 0.05',)
plt.xlabel('Iterations')
plt.ylabel('Error wrt to optimal weights')
plt.plot(its,error_plms, label = 'PLMS')
plt.plot(its,error_lms, label = 'LMS' )
plt.plot(its,error_rls, label = 'RLS' )
plt.legend()
plt.show()

#Output w/o noise
w = np.random.normal(0, 1, X.shape[1])
y = np.zeros(X.shape[0])
for i in range(0, X.shape[0]):
    y[i] = np.dot(X[i,:], w)
    w = np.random.multivariate_normal(mean=w, cov=sigma_diffuse*np.identity(X.shape[1]), size=1).flatten()
#Adding noise
v = np.random.normal(0, sigma_noise, len(y))

#Training PLMS Filter
solver = PLMS_solver(X,y + v,sigma_noise, sigma_diffuse)
solver.Learn()
error_plms = solver.TrackOutput()

#Training RLS Filter
solver = RLS_solver(X,y + v,sigma_noise, sigma_diffuse)
solver.Learn()
error_rls = solver.StationaryError(w_opt)

#Training LMS Filter
solver = LMS_solver(X,y + v,sigma_noise, sigma_diffuse)
solver.Learn()
error_lms = solver.TrackOutput()

its = np.arange(X.shape[0])
plt.title('NonStationary, noise = 0.1,diffusion = 0.05',)
plt.xlabel('Iterations')
plt.ylabel('Tracking Error')
plt.plot(its, (error_plms - y), label = 'PLMS')
plt.plot(its, (error_lms - y), label = 'LMS')
plt.plot(its,(error_rls - y), label = 'RLS' )
plt.legend()
plt.show()
