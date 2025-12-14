import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from PIL import Image

class Env():
    """
    Environment wrapper for CarRacing
    """
    def __init__(self, record_video=False, final_epoch=False, folder_name="epochs_", img_stack=4, action_repeat=8):
        # Initial environment creation to get thresholds and basic properties
        self.env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
        self.reward_threshold = self.env.spec.reward_threshold
        
        if record_video:
            if final_epoch:
                name_prefix="eval_final_episode"
            else:
                name_prefix="eval_moving_avg_episode"
            self.env = RecordVideo(
                gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True),
                video_folder=folder_name+"/video", name_prefix=name_prefix,
                episode_trigger=lambda x: True
            )
        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        self.die = False
        img_rgb = self.env.reset()[0]
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):   #speed up training
            img_rgb, reward, die, _, _ = self.env.step(action)
            total_reward += reward
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)      
        return np.array(self.stack), total_reward, done, die

    def rgb2gray(self, rgb):
        img_pil = Image.fromarray(rgb)
        gray = np.array(img_pil.convert("L"))  # "L" mode for grayscale
        gray = gray / 128.0 - 1.0
        return gray

    def reward_memory(self):
        # record reward for last 100 steps
        self.count = 0
        self.length = 100
        self.history = np.zeros(self.length)

        def memory(reward):
            self.history[self.count] = reward
            self.count = (self.count + 1) % self.length
            return np.mean(self.history)

        return memory
