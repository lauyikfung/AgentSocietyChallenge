# Recommendation Agent

- The agent is stored in myagent2.py
- In "AgentSocietyChallenge/websocietysimulator/llm/llm.py", MyLLM class is implemented for SGLang-style API call
  - To start SGLang API service, first pip install "sglang", then setup sglang model server: `python3 -m sglang.launch_server --model-path Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --log-level warning --port 12345 --dp-size 4` (you can change dp-size for multiple GPUs)
- Then you can run the following command for testing recommendation task:
- ```
  from websocietysimulator import Simulator
  from websocietysimulator.llm import LLMBase, MyLLM
  from example.myagent2 import MyRecommendationAgent
  import os
  # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
  # Initialize Simulator
  simulator = Simulator(data_dir="./processed_data", device="auto", cache=False)
  # The cache parameter controls whether to use cache for interaction tool.
  # If you want to use cache, you can set cache=True. When using cache, the simulator will only load data into memory when it is needed, which saves a lot of memory.
  # If you want to use normal interaction tool, you can set cache=False. Notice that, normal interaction tool will load all data into memory at the beginning, which needs a lot of memory (20GB+).

  task_set = "amazon" # "goodreads" or "yelp"
  # Load scenarios
  simulator.set_task_and_groundtruth(task_dir=f"./example/track2/{task_set}/tasks", groundtruth_dir=f"./example/track2/{task_set}/groundtruth")

  # Set your custom agent
  simulator.set_agent(MyRecommendationAgent)

  # Set LLM client
  simulator.set_llm(MyLLM(port="12345"))

  # Run evaluation
  # If you don't set the number of tasks, the simulator will run all tasks.
  agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

  # Evaluate the agent
  evaluation_results = simulator.evaluate()
  ```
