import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from agent import PPOAgent
from environment import Env

def train_agent(epochs):
    # Initialize the folder name based on the number of epochs
    folder_name = f"checkpoints/param_{epochs}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Initialize the agent and environment
    # Note: Ensure log directory exists or Update PPOAgent to handle it
    agent = PPOAgent(folder_name=folder_name, epochs=epochs)
    env = Env(folder_name=folder_name)

    running_score = 0
    state = env.reset()
    log_interval = 10

    # Training Loop
    print(f"Starting training for {epochs} epochs...")
    for i_ep in tqdm(range(epochs)):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

            # Store the transition and update agent if buffer is full
            if agent.store((state, action, a_logp, reward, state_)):
                # print('updating') # verbose
                agent.update()

            # Update score and state for next time step
            score += reward
            state = state_

            # Break the loop if the episode ends
            if done or die:
                break

        # Update running score with exponential decay
        running_score = running_score * 0.99 + score * 0.01

        # Log episode score and moving average to TensorBoard
        agent.writer.add_scalar("Episode/Score", score, i_ep)  # Log score for this episode
        agent.writer.add_scalar("Episode/MovingAverageScore", running_score, i_ep)  # Log moving average

        # Save model with the best moving average score
        agent.save_max_moving_average_param(running_score)

        if i_ep % log_interval == 0:
            # Save logs to a text file
            with open(f"logs/logs_param_{epochs}.txt", "a") as f:
                f.write('Ep {}\\tLast score: {:.2f}\\tMoving average score: {:.2f}\\n'.format(i_ep, score, running_score))

            # Save model parameters
            agent.save_param()

            # Save model every 500 episodes (or 10 as in original code?) -> original was 10
            if i_ep % 50 == 0: 
                agent.save_param(i_ep)

        # If running score exceeds reward threshold, stop training
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            agent.save_param()
            break

    # After training, print the paths to the saved parameter files
    param_file_path = ''    
    max_average_file_path = ''

    for file in glob.glob(f'{folder_name}/*'):
        if 'max' in file: max_average_file_path = os.path.abspath(file)
        else: param_file_path = os.path.abspath(file)

    print(f"Training finished.")
    print(f"Param file path: {param_file_path}")
    print(f"Max average file path: {max_average_file_path}")

    return param_file_path, max_average_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent for CarRacing-v2")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    train_agent(args.epochs)
