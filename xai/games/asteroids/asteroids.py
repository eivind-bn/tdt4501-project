from typing import *
from ale_py import ALEInterface, ALEState
from random import random

import gymnasium as gym
import cv2

from ..env import Environment
from .observation import AsteroidsObservation
from .reward import AsteroidsReward
from .action import AsteroidsAction


class Asteroids(Environment[AsteroidsObservation,AsteroidsAction,AsteroidsReward]):

    def __init__(self) -> None:
        
        self._ale: ALEInterface = gym.make("Asteroids-v4")\
            .get_wrapper_attr("ale")
        
        height, width = self._ale.getScreenDims()
        observation_shape = (height, width, 3)
        super().__init__(observation_shape=observation_shape)
        self.reset()

    def step(self, 
             action:        AsteroidsAction,
             stochastic:    bool = False,
             steps:         int = 1) -> AsteroidsReward:
        
        native_rewards: List[int|float] = []
        images: List[AsteroidsObservation] = []

        for _ in range(steps):
            reward = 0
            if stochastic:
                if random() < 0.5:
                    reward += self._step_spaceship(AsteroidsAction.NOOP)
                else:
                    reward += self._step_asteroids(AsteroidsAction.NOOP)

            reward += self._step_spaceship(action)
            reward += self._step_asteroids(action)

            native_rewards.append(reward)
            images.append(AsteroidsObservation(
                spaceship=self.spaceship,
                asteroids=self.asteroids,
                spaceship_angle=self.get_angle()
            ))
        
        return AsteroidsReward(
            values=native_rewards,
            observations=images,
            actions=[action]*steps
        )

    def render(self) -> AsteroidsObservation:
        return AsteroidsObservation(
            spaceship=self.spaceship, 
            asteroids=self.asteroids,
            spaceship_angle=self.get_angle()
            )
    
    def _step_asteroids(self, action: AsteroidsAction) -> int:
        flags = int(self._ale.getRAM()[57])
        self._ale.setRAM(57,~1&flags)
        reward = self._ale.act(action.value)
        self.asteroids = self._ale.getScreenRGB()
        return reward
    
    def _step_spaceship(self, action: AsteroidsAction) -> int:
        flags = int(self._ale.getRAM()[57])
        self._ale.setRAM(57,1|flags)
        reward = self._ale.act(action.value)
        self.spaceship = self._ale.getScreenRGB()
        return reward
        
    def running(self) -> bool:
        return not self._ale.game_over()
    
    def lives(self) -> int:
        return self._ale.lives()
    
    def get_angle(self) -> float:
        angle_step = self._ale.getRAM()[60] & 0xf
        return self._angle_steps_to_radians[angle_step]

    def reset(self) -> None:
        self._ale.reset_game()
        self._angle_steps_to_radians = (
            0.0, 
            0.23186466084938862, 
            0.5880026035475675, 
            0.9037239459029813, 
            1.5707963267948966, 
            2.256525837701183, 
            2.6909313275091598, 
            2.936197264400026, 
            3.141592653589793, 
            3.2834897081939567, 
            3.597664649939404, 
            4.023464592169828, 
            4.71238898038469, 
            5.365235611485464, 
            5.81953769817878, 
            6.120457932539206
        )
        assert (sum(self._angle_steps_to_radians) - 47.24187379318632) < 1e-4
        self.step(AsteroidsAction.NOOP)
    
    def clone_state(self) -> ALEState:
        return self._ale.cloneState()
    
    def restore_state(self, state: ALEState) -> None:
        self._ale.restoreState(state)

    def play(self,
             fps:           int = 60,
             scale:         float = 4.0,
             translate:     bool = False,
             rotate:        bool = False,
             stochastic:    bool = False,
             step_cb:       Callable[[AsteroidsObservation,AsteroidsReward],None] = lambda *_: None) -> None:
        
        title = "Asteroids"

        def window_visible() -> bool:
            return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) > 0
        
        def wait_key() -> str|None:
            code = cv2.waitKeyEx(1000//fps)
            if code > -1:
                return chr(code)
            else:
                return None  

        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)

            h,w,_ = self.observation_shape
            cv2.resizeWindow(title, int(w*scale), int(h*scale))

            while self.running() and window_visible():
                
                image = self.render()

                if translate:
                    image = image.translated()

                if rotate:
                    image = image.rotated()
                
                cv2.imshow(title, image.numpy()[:,:,::-1])
                
                key = wait_key()

                match key:
                    case "q":
                        break
                    case "w":
                        step_cb(image, self.step(AsteroidsAction.UP, stochastic=stochastic))
                    case "a":
                        step_cb(image, self.step(AsteroidsAction.LEFT, stochastic=stochastic))
                    case "d":
                        step_cb(image, self.step(AsteroidsAction.RIGHT, stochastic=stochastic))
                    case " ":
                        step_cb(image, self.step(AsteroidsAction.FIRE, stochastic=stochastic))
                    case _:
                        step_cb(image, self.step(AsteroidsAction.NOOP, stochastic=stochastic))

        finally:
            cv2.destroyWindow(title)
