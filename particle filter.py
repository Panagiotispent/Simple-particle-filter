# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:01:37 2020

@author: panay
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import random
from numpy.random import seed
from numpy.random import uniform
import scipy.stats


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty(shape=(N,3))
    particles[:,0] = uniform(x_range[0], x_range[1], size=N)
    particles[:,1] = uniform(y_range[0], y_range[1], size=N)
    particles[:,2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:,2] %= 2*np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N,3))
    particles[:,0] = mean[0]+(randn(N)*std[0])
    particles[:,1] = mean[1]+(randn(N)*std[1])
    particles[:,2] = mean[2]+(randn(N)*std[2])
    particles[:,2]%=2*np.pi
    return particles

def predict(particles, u, std, dt=1):
    """ move according to control input u (heading change, velocity) with noise Q (std heading change, std velocity)`"""
    # In this example we received u = (0, 1.44) which doesn't change the direction, only makes move forward.
    N = len(particles)
    
    # update heading, third column
    particles[:,2] += u[0] + (randn(N) * std[0])
    particles[:,2] %= 2 * np.pi
    
    # move in the (noisy) commanded direction
    dist = (u[1]*dt) + (randn(N)*std[1])
    
    # We take the direction regarding the X axis taking the cosinus, then we multiply the distance with some noise
    # We add this result to the X values of the particles to make them move forward
    particles[:,0] += np.cos(particles[:,2]) * dist
    # We take the direction regarding the Y axis taking the sinus, then we multiply the d istance with some noise
    # We add this result to the Y values of the particles to make them move forward
    particles[:,1] += np.sin(particles[:,2]) * dist
    
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        # Give the distance between each of our particules with the landmark
        distance = np.linalg.norm(particles[:,0:2] - landmark, axis=1)
        # We create a gaussian of mean "distance", std as R, and then look at the probability to obtain z
        alpha = scipy.stats.norm(distance, R).pdf(z[i])
        weights *= alpha #donc W_t^i = W_{t-1}^i * alpha            
    
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)# normalize   
        
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    pos = particles[:,0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos-mean)**2, weights=weights, axis=0)
    return mean, var 

def neff(weights):
    return 1./np.sum(np.square(weights))  

def systematic_resample(weights):
    N = len(weights)
    
    # We create a tab of of values going from zero to the weights' length, we add to each value a random number.
    # We normalize be dividing with the number of weights
    # "positions" is only composed of low values (because normalized) increasing lightly
    # The latter is logical since their're composed of their initial index (ex: [1,2,3] --> [0.0006671 0.0016671 0.0026671])
    positions = (random() + np.arange(N)) / N
    
    # We create indexes with the same length than weights, having only 0's
    indexes = np.zeros(N, 'i')
    
    # cumucumulative_sum is a tab equal to the cumulative sum of the weights ...
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    # i is going to help us to parse all the values to change onto j
    # j is the index of the value that we want to attribute to the ith value
    # with cumulative_sum, we're going to look for a value where the weight would
    # significatly increase the cumulative sum, letting the latter be higher than positions[i]
    # While we don't find a weight value increasing the cumulative sum above the position[i], we don't modify i
    # It's possible that we would have to increase j a lot before having i being equal to 1 (if a lot of weights a low).
    while i < N:
        if positions[i] < cumulative_sum[j]:
            # If we have a big weight, then we attribute the latter's index to your index i
            indexes[i] = j
            # Then we increase i to give values to other indexes. Maybe we might give this index j to i+1
            i += 1
        else:
            # While we don't have any significant weight, we increase j.
            j += 1
       
        # Basically it returns the particules indexes that we will duplicate (with high weight values)
        # Thus, when we'll give these indexes to the particules & weights tabs, their're going to be replicated.
    return indexes     

def resample_from_index(particles, weights, indexes):
    # Indexes has the same length as particules, it has all the indexes of the particules with high weights.
    # example: indexes = [3, 3, 3, 18, 21]. (here we multiply the particule[3] by 3)
    # Applying indexes to particules & weights, will replicated the indexed values
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0/len(weights)) 
    
def run_particle_filter(N, iters=18, sensor_std_err=.1, do_plot=True, plot_particles=False, xlim=(0,20), ylim=(0,20), initial_x=None):
    
    landmarks=np.array([[-1,2], [5,10], [12,14], [18,21]])
    NL=len(landmarks)
    plt.figure(figsize=(15,15))
    plt.figure()
    
    # create particles and weights,
    # If we know the initial position, we can specify this latter as the mean of a gaussian
    # to generate samples from that information
    if initial_x is not None:
        particles = create_gaussian_particles( mean=initial_x, std=(5,5, np.pi/4), N=N)
    else:
        particles = create_uniform_particles((0,20), (0,20), (0,6.28), N)
    weights=np.zeros(N)

    if plot_particles:
        alpha =.20
        if N>5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)
        plt.scatter(particles[:,0], particles[:,1],alpha=alpha, color='g')
    
    xs=[]
    robot_pos=np.array([0.,0.])

    for x in range(iters):
        robot_pos+=(1,1)
        
        # distance from robot to each landmark
        zs=(norm(landmarks-robot_pos, axis=1)+(randn(NL)*sensor_std_err))
        
        # We move the particules diagonally and uniformly forward to (x+1, x+1)
        predict(particles, u=(0.00,1.414), std=(.2,.05))
        
        # After our move, we now need to update our weights
        # We update the weights with zs --> incorporate measurements
        # Update() doesn't change the particules positions, just their weight values regarding the obtained zs
        update(particles, weights, z=zs, R=sensor_std_err,landmarks=landmarks)

        # resample if too few effective particles
        if neff(weights) < N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
        
        # Compute our estimation
        mu, var=estimate(particles, weights)
        
        xs.append(mu)
        
        if plot_particles:
            plt.scatter(particles[:,0], particles[:,1],color='k', marker=',', s=1)

        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
    
    xs=np.array(xs)
    
    plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual','PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu-np.array([iters, iters]), var)
    plt.show()
    
if __name__ == '__main__':
    #seed(2)
    #run_particle_filter(N=5000, plot_particles=False)
    #run_particle_filter(N=5000, iters=8, plot_particles=True,xlim=(0,9), ylim=(0,9))
    
    seed(6)
    run_particle_filter(N=5000, plot_particles=True, initial_x=(1,1, np.pi/4))
    
    
    
    
    