#!/usr/bin/env python
import traceback

import gym
import numpy as np
from collections import defaultdict

# noinspection PyUnresolvedReferences
import gym_duckietown_agent  # DO NOT CHANGE THIS IMPORT (the environments are defined here)
from duckietown_challenges import wrap_solution, ChallengeSolution, ChallengeInterfaceSolution, InvalidEnvironment
from wrappers import SteeringToWheelVelWrapper
from PIL import Image, ImageDraw


def launch_env(id=None):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123, # random seed
            map_name="loop_empty",
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env


def launch_local_experiment(environment):
    # Use our launcher
    env = launch_env(environment)

    # === BEGIN SUBMISSION ===

    # If you created custom wrappers, you also need to copy them into this folder.

    from wrappers import NormalizeWrapper, ImgWrapper, ActionWrapper, ResizeWrapper

    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    # to make the images pytorch-conv-compatible
    env = ImgWrapper(env)
    env = ActionWrapper(env)

    # you ONLY need this wrapper if you trained your policy on [speed,steering angle]
    # instead [left speed, right speed]
    # env = SteeringToWheelVelWrapper(env)

    # you have to make sure that you're wrapping at least the actions
    # and observations in the same as during training so that your model
    # receives the same kind of input, because that's what it's trained for
    # (for example if your model is trained on grayscale images and here
    # you _don't_ make it grayscale too, then your model wont work)

    # HERE YOU NEED TO CREATE THE POLICY NETWORK SAME AS YOU DID IN THE TRAINING CODE
    # if you aren't using the DDPG baseline code, then make sure to copy your model
    # into the model.py file and that it has a model.predict(state) method.
    import model

    my_model = model.model()

    # === END SUBMISSION ===

    # deactivate the automatic differentiation (i.e. the autograd engine, for calculating gradients)
    observation = env.reset()

    # While there are no signal of completion (simulation done)
    # we run the predictions for a number of episodes, don't worry, we have the control on this part
    trial = 0
    reward =.0000001
    rewards = defaultdict(list)
    while trial < 1:
        trial += 1
        step = 0
        while reward != -1000 and step != 300:
            step += 1
            # we passe the observation to our model, and we get an action in return
            action = my_model.predict(observation, trial, step)
            print("step: "+str(step))
            print(action)
            # we tell the environment to perform this action and we get some info back in OpenAI Gym style
            observation, reward, done, info = env.step(action)
            # here you may want to compute some stats, like how much reward are you getting
            # notice, this reward may no be associated with the challenge score.
            rewards[trial].append(reward)
            # it is important to check for this flag, the Evalution Engine will let us know when should we finish
            # if we are not careful with this the Evaluation Engine will kill our container and we will get no score
            # from this submission
        print("num_steps: " + str(float(len(rewards[trial]))))
        print("mean reward: " + str(sum(rewards[trial]) / float(len(rewards[trial]))))
        env.reset()
        reward = 0

if __name__ == '__main__':
    print('Starting submission')
    launch_local_experiment(environment=None)
