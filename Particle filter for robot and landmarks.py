#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:49:06 2020

@author: https://urban-analytics.github.io/dust/2019/05/ParticleFilter-BlogPost-KM.html
"""

from numpy.random import uniform
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from numpy.random import seed

def create_uniform_particles(x_range, y_range, hdg_range, num_of_particles):
    '''
    Create Uniformly Distributed Particles
    
    PARAMETERS
     - x_range:             Interval of x values for particle locations   
     - y_range:             Interval of y values for particle locations 
     - hdg_range:           Interval of heading values for particles in radians
     - num_of_particles:    Number of particles
    
    DESCRIPTION
    Create N by 3 array to store x location, y location, and heading
    of each particle uniformly distributed. Take modulus of heading to ensure heading is 
    in range [0,2*pi).
    
    Returns particle locations and headings
    '''
    particles = np.empty((num_of_particles, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size = num_of_particles)
    particles[:, 1] = uniform(y_range[0], y_range[1], size = num_of_particles)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size = num_of_particles)
    particles[:, 2] %= 2 * np.pi
    return particles

def predict(particles, control_input, std, dt=1.):
    '''
    Predict particles
    
    PARAMETERS
     - particles:       Locations and headings of all particles
     - control_input:   Heading, x location step, and y location step to predict particles
     - std:             Standard deviation for noise added to prediction
     - dt:              Time step
     
    DESCRIPTION
    Predict particles forward one time step using control input u
    (heading, x step, y step) and noise with standard deviation std.
    '''
    num_of_particles = len(particles)
    
    # update heading
    particles[:, 2] += control_input[0] + (randn(num_of_particles) * std[0])
    particles[:, 2] %= 2 * np.pi

    # calcualte change in x and y directions
    xdist = (control_input[1] * dt) + (randn(num_of_particles) * std[1])
    ydist = (control_input[2] * dt) + (randn(num_of_particles) * std[2])
    
    # add changes to current x and y locations
    particles[:, 0] += xdist
    particles[:, 1] += ydist
    
    # uncomment below when u = (heading, velocity) to predict
    # particles using a velocity rather than an x and y change.
    #
    #dist = (u[1] * dt) + (randn(num_of_particles) * std[1])
    #particles[:, 0] += np.cos(particles[:,2])*dist
    #particles[:, 1] += np.sin(particles[:,2])*dist
    
    
def update(particles, weights, observation, sensor_std, landmarks):
    '''
    Update particle weights
    
    PARAMETERS
     - particles:    Locations and headings of all particles
     - weights:      Weights of all particles
     - observation:  Observation of distances between robot and all landmarks
     - sensor_std:   Standard deviation for error in sensor for observation
     - landmarks:    Locations of all landmarks
    
    DESCRIPTION
    Set all weights to 1. For each landmark, calculate the distance between 
    the particles and that landmark. Then, for a normal distribution with mean 
    = distance and std = sensor_std, calculate the pdf for a measurement of observation. 
    Multiply weight by pdf. If observation is close to distance, then the 
    particle is similar to the true state of the model so the pdf is close 
    to one so the weight stays near one. If observation is far from distance,
    then the particle is not similar to the true state of the model so the 
    pdf is close to zero so the weight becomes very small.   
    
    The distance variable depends on the particles while the z parameter depends 
    on the robot.
    '''
    weights.fill(1.)
    
    for i, landmark in enumerate(landmarks):
        
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, sensor_std).pdf(observation[i])
        
    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize
    
def effective_n(weights):
    '''
    Calculate effective N
    
    PARAMETERS
     - weights:    Weights of all particles
    
    DESCRIPTION
    Calculates effective N, an approximation for the number of particles 
    contributing meaningful information determined by their weight. When 
    this number is small, the particle filter should resample the particles
    to redistribute the weights among more particles.
    
    Returns effective N.
    '''
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    '''
    Resample particles by index
    
    PARAMETERS
     - particles:    Locations and headings of all particles
     - weights:      Weights of all particles
     - indexes:      Indexes of particles to be resampled
     
    DESCRIPTION
    Resample particles and the associated weights using indexes. Reset
    weights to 1/N = 1/len(weights).
    '''
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill (1.0 / len(weights))
    
def estimate(particles, weights):
    '''
    Estimate state of system
    
    PARAMETERS:
     - particles:    Locations and headings of all particles
     - weights:      Weights of all particles
    
    DESCRIPTION
    Estimate the state of the system by calculating the mean and variance
    of the weighted particles.
    
    Returns mean and variance of the particle locations
    '''
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def run_textbook_particle_filter(num_of_particles, 
                        num_of_iterations = 20, 
                        sensor_std = .1, 
                        do_plot = True, 
                        plot_particles = False):
    '''
    Run the particle filter
    
    PARAMETERS
     - num_of_particles:     Number of particles
     - num_of_iterations:    Number of iterations for particle filter
     - sensor_std:           Standard deviation for error of sensor
     - do_plot:              Boolean variable to plot particle filter results
     - plot_particles:       Boolean variable to plot each particle
    
    DESCRIPTION
    Set locations for landmarks, particle locations, and particle weights.
    Plot individual particles initially. Set robot location. For each
    iteration, increment the robot location, take observation with noise 
    for distance between robot and landmarks, predict particles forward, 
    update particle weights. If effective N is small enough, resample
    particles. Calculate estimates and save the particle mean. Plot 
    results and print final error statistics.
    '''
    
    landmarks = np.array([[-1, 0], [2, 3], [-1,15], [2,36]])
    num_of_landmarks = len(landmarks)
    
    plt.figure()
   
    # create particles
    particles = create_uniform_particles((0,20), (0,20), (0, 6.28), num_of_particles)
    weights = np.zeros(num_of_particles)

    # plot particles
    if plot_particles:
        alpha = .20
        if num_of_particles > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(num_of_particles)           
        plt.scatter(particles[:, 0], particles[:, 1], alpha=alpha, color='g')
    
    means = []
    robot_position = np.array([0., 0.])
    
    # loop through iterations
    for iteration in range(num_of_iterations):
        
        # increment robot location
        robot_position += (1, 1)
    
        # distance from robot to each landmark
        observation = (norm(landmarks - robot_position, axis=1) + 
              (randn(num_of_landmarks) * sensor_std))
        
        # move diagonally forward
        predict(particles, control_input=(0.00, 1., 1.), std=(.2, 5, 5))
        
        # incorporate measurements
        update(particles, weights, observation=observation, sensor_std=sensor_std, 
               landmarks=landmarks)
        
        # resample if too few effective particles
        if effective_n(weights) < num_of_particles/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)

        # calculate and save estimates
        mean, variance = estimate(particles, weights)
        means.append(mean)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1], 
                        color='k', marker=',', s=1)
        p1 = plt.scatter(robot_position[0], robot_position[1], marker='+',
                         color='k', s=180, lw=3)
        p2 = plt.scatter(mean[0], mean[1], marker='s', color='r')
    
    means = np.array(means)
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    print('final position error, variance:\n\t', 
          mean - np.array([num_of_iterations, num_of_iterations]), variance)
    plt.show()

seed(2) 
run_textbook_particle_filter(num_of_particles=5000, plot_particles=False)