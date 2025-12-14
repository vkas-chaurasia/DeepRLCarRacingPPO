import numpy as np
import argparse
from tqdm import tqdm
from agent import PPOAgent, TestPPOAgent
from environment import Env

def test_agent(param_file_path, max_average_file_path, folder_name, final_epoch_testing=False, total_runs=1, agent=None):
    # Initialize the test agent and environment
    test_agent = TestPPOAgent(param_file_path=param_file_path, max_average_file_path=max_average_file_path, final_epoch_testing=final_epoch_testing)
    env = Env(record_video=True, folder_name=folder_name)
    
    scores = []
    state = env.reset()

    print(f"Starting testing for {total_runs} runs...")
    # Testing Loop
    for i_ep in tqdm(range(total_runs)):
        score = 0
        state = env.reset()
        for t in range(1000):
            action = test_agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            score += reward
            state = state_
            if done or die:
                break
        
        # Log score for this episode if agent is provided (for tensorboard)
        if agent:
            if final_epoch_testing:
                agent.writer.add_scalar("TestEpisode/Score", score, i_ep)
            else:
                agent.writer.add_scalar("TestEpisode/MovingAverageScore", score, i_ep)
        
        print('Ep {}\\tScore: {:.2f}\\t'.format(i_ep, score))
        scores.append(score)

    # Print summary statistics
    print('Average Score:', np.mean(scores))
    print(f"Maximum score: {np.max(scores)}")
    print(f"Minimum score: {np.min(scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PPO Agent for CarRacing-v2")
    parser.add_argument("--param_path", type=str, required=True, help="Path to the parameter file")
    parser.add_argument("--max_avg_path", type=str, default="", help="Path to the max moving average parameter file (optional)")
    parser.add_argument("--folder_name", type=str, default="test_results", help="Folder to save videos")
    parser.add_argument("--runs", type=int, default=5, help="Number of test runs")
    parser.add_argument("--final", action="store_true", help="Use final epoch parameters instead of max average")

    args = parser.parse_args()
    
    # If max_avg_path is not provided, use param_path for both or handle accordingly
    # The logic in TestPPOAgent uses max_average_file_path primarily, and param_file_path if final_epoch_testing is set.
    # So we pass args.param_path as max_average if checking max average, or as param_path if final.
    
    if args.final:
        # User wants to test specific epoch params
        # max_avg can be dummy if not used
        test_agent(param_file_path=args.param_path, max_average_file_path=args.max_avg_path, folder_name=args.folder_name, final_epoch_testing=True, total_runs=args.runs)
    else:
        # User wants to test best model
        # Logic expects params in max_average_file_path
        path_to_use = args.max_avg_path if args.max_avg_path else args.param_path
        test_agent(param_file_path="", max_average_file_path=path_to_use, folder_name=args.folder_name, final_epoch_testing=False, total_runs=args.runs)
