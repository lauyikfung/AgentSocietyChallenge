import json
import re
import ast
import logging
import tiktoken
import os
import threading  # Added for thread-safe memory
from collections import Counter

# --- Imports from Provided Files ---
# (Assuming these are available in the environment)
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent

# --- from reasoning_modules.py ---
# This class is unchanged, as it only uses the LLM API
class ReasoningBase:
    def __init__(self, profile_type_prompt, memory, llm):
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm = llm
    
    def process_task_description(self, task_description):
        examples = ''
        return examples, task_description

class ReasoningSelfRefine(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        
        # 1. Initial Reasoning
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        
        # 2. Self-Refinement Step
        refined_result = self.refine(reasoning_result)
        return refined_result

    def refine(self, reasoning_result):
        # The refinement prompt asks the LLM to critique its own work
        prompt = f'''Reflect on the following reasoning process and the resulting ranked list. 
Identify any potential errors, contradictions, or areas for improvement based on the user's profile.
For example, does the ranking align with the user's historical likes and dislikes?
Provide a revised version of the reasoning and the final ranked list. If the original reasoning is high-quality, you may state that and return it.

Here is the original reasoning and ranking:
{reasoning_result}

Critique and Revised Output:
'''     
        messages = [{"role": "user", "content": prompt}]
        feedback_result = self.llm(
            messages=messages,
            temperature=0.0 # Use temp 0 for the refinement step
        )
        return feedback_result


# --- New Module: JSON User Preference Memory ---
# This class replaces the Chroma-based memory.
# It requires NO embedding model.

class JSONUserPreferenceMemory:
    """
    A thread-safe, file-based memory module to store and retrieve user preference
    summaries. It uses a simple JSON file as a key-value store.
    """
    def __init__(self, llm, db_path='./db/api_memory'):
        self.llm = llm  # Keep LLM for the reflection step
        self.memory_file = os.path.join(db_path, 'user_prefs.json')
        self.lock = threading.Lock()  # To handle concurrent writes from threads
        self.memory_data = self._load_memory()
        
    def _load_memory(self) -> dict:
        """Loads the memory file from disk in a thread-safe way."""
        with self.lock:
            try:
                if os.path.exists(self.memory_file):
                    with open(self.memory_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load memory file, starting fresh: {e}")
            return {}

    def _save_memory(self):
        """Saves the in-memory dict to disk in a thread-safe way."""
        with self.lock:
            try:
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(self.memory_data, f, indent=4)
            except IOError as e:
                logging.error(f"Could not save memory file: {e}")

    def retriveMemory(self, user_id: str) -> str:
        """
        Retrieves a user's preference summary.
        """
        # No lock needed for read, as self.memory_data is only *replaced*
        # in _load_memory, but we'll use it for safety with get().
        with self.lock:
            return self.memory_data.get(user_id, "")

    def reflect_and_add_memory(self, user_id: str, user_summary: str, reviews_summary: str):
        """
        Uses the LLM to reflect on user data and saves the resulting
        preference summary to the JSON file.
        """
        try:
            # 1. Reflect: Use LLM to generate the summary
            memory_prompt = f"""
Based on the following user data, generate a concise summary of the user's preferences (likes and dislikes) to be used for future recommendations. Focus on strong signals.

User Profile: {user_summary}
User History: {reviews_summary}

Concise Preference Summary:
"""
            new_memory_summary = self.llm(
                messages=[{"role": "user", "content": memory_prompt}],
                temperature=0.0,
                max_tokens=500
            )

            # 2. Add & Save: Update the dictionary and save to file
            with self.lock:
                self.memory_data[user_id] = new_memory_summary.strip()
            
            self._save_memory()
            logging.info(f"[{user_id}] Memory updated.")

        except Exception as e:
            logging.warning(f"[{user_id}] Failed to update memory: {e}")


# --- New Module: APIBasedRefiningAgent ---

class APIBasedRefiningAgent(RecommendationAgent):
    """
    An advanced recommendation agent that uses:
    1.  JSONUserPreferenceMemory: For embedding-free long-term preference storage.
    2.  LLM Summarization: To intelligently compress context.
    3.  ReasoningSelfRefine: To perform Chain-of-Thought ranking and 
        immediate self-critique.
    """
    
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        
        # Initialize the advanced reasoning module
        self.reasoning = ReasoningSelfRefine(
            profile_type_prompt='', 
            memory=None,  # Memory is handled manually
            llm=self.llm
        )
        
        # Initialize the new JSON-based memory module
        self.memory = JSONUserPreferenceMemory(llm=self.llm)
        
        # Setup token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _num_tokens(self, string: str) -> int:
        """Helper to count tokens."""
        return len(self.encoding.encode(string))

    def _llm_summarize(self, text: str, max_tokens: int = 1500, focus: str = "user preferences, likes, and dislikes") -> str:
        """
        Summarizes long text using the LLM to stay within context limits.
        This is "smart truncation".
        """
        if self._num_tokens(text) <= max_tokens:
            return text
        
        prompt = f"""Summarize the following text to be concise (under {max_tokens} tokens). 
Focus specifically on details relevant to {focus}.

Text to summarize:
{text}

Concise Summary:
"""
        try:
            summary = self.llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens
            )
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            # Fallback to naive truncation if LLM fails
            return self.encoding.decode(self.encoding.encode(text)[:max_tokens])

    def _parse_list(self, llm_output: str) -> list[str]:
        """
        Robustly parses the final ranked list from the LLM's output.
        """
        try:
            # Find the last occurrence of a Python-like list
            match = re.search(r"\[\s*('|\").*?('|\")\s*\]", llm_output, re.DOTALL)
            if match:
                list_str = match.group(0)
                # Use ast.literal_eval for safe parsing
                return ast.literal_eval(list_str)
            else:
                # print(f"Parsing Error: No list found in output. Output: {llm_output[:200]}...")
                print(f"Parsing Error: No list found in output. Output: {llm_output}")
                return ['']
        except Exception as e:
            print(f"Parsing Error: {e}. Output was: {llm_output}")
            return [''] # Return empty on failure

    def workflow(self):
        """
        Executes the full, multi-step recommendation workflow.
        """
        user_id = self.task['user_id']
        candidate_list = self.task['candidate_list']
        
        # --- 1. Planning & Memory Retrieval ---
        # Retrieve long-term memory for this user, if any exists.
        logging.info(f"[{user_id}] Step 1: Retrieving User Memory")
        memory_summary = self.memory.retriveMemory(user_id)
        if memory_summary:
            logging.info(f"[{user_id}] Found memory: {memory_summary[:100]}...")
        else:
            logging.info(f"[{user_id}] No prior memory found.")
            
        # --- 2. Context Retrieval ---
        logging.info(f"[{user_id}] Step 2: Gathering Context")
        user_data = str(self.interaction_tool.get_user(user_id=user_id))
        
        item_data_list = []
        for item_id in candidate_list:
            item = self.interaction_tool.get_item(item_id=item_id)
            keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
            filtered_item = {key: item[key] for key in keys_to_extract if key in item}
            item_data_list.append(filtered_item)
        
        history_review_data = str(self.interaction_tool.get_reviews(user_id=user_id))

        # --- 3. Smart Context Summarization ---
        logging.info(f"[{user_id}] Step 3: Summarizing Context")
        user_summary = self._llm_summarize(
            user_data, 
            max_tokens=1000, 
            focus="user's profile, demographics, and stated interests"
        )
        items_summary = self._llm_summarize(
            str(item_data_list), 
            max_tokens=4000, 
            focus="item attributes, average ratings, and review counts"
        )
        reviews_summary = self._llm_summarize(
            history_review_data, 
            max_tokens=4000, 
            focus="user's historical likes/dislikes, sentiments, and star ratings"
        )

        # --- 4. Reasoning & Self-Refinement ---
        logging.info(f"[{user_id}] Step 4: Reasoning & Self-Refinement")
        
        task_description = f"""
You are a recommendation expert. Your task is to rank a list of 20 candidate items for a user.
You must provide a step-by-step chain of thought for your reasoning *before* giving the final list.

--- 1. Prior Preference Summary (from past tasks) ---
{memory_summary if memory_summary else "No prior preferences on file."}

--- 2. Current User Profile (Summary) ---
{user_summary}

--- 3. User's Historical Reviews (Summary) ---
{reviews_summary}

--- 4. Candidate Items (Summary) ---
{items_summary}

--- TASK ---
Analyze all the information above. 
1.  First, build a detailed model of the user's preferences (likes and dislikes) from their profile and review history.
2.  Second, analyze the candidate items.
3.  Third, compare the user's preferences to the items and explain your ranking logic step-by-step.
4.  Finally, output ONLY the ranked list of item IDs in the specified format.

The candidate list to rank is: {candidate_list}

Your output format MUST be:

Thought:
[Your detailed, step-by-step reasoning and analysis goes here.]

Final Ranked List:
['item_id1', 'item_id2', 'item_id3', ...]
"""
        
        final_output = self.reasoning(task_description)
        final_ranking = self._parse_list(final_output)

        # --- 5. Reflection & Memory Update ---
        # Generate and save an updated preference summary for next time.
        logging.info(f"[{user_id}] Step 5: Reflecting & Updating Memory")
        # This one call now handles both reflection (LLM call) and saving
        self.memory.reflect_and_add_memory(
            user_id=user_id,
            user_summary=user_summary,
            reviews_summary=reviews_summary
        )

        # --- 6. Return Final Ranking ---
        logging.info(f"[{user_id}] Task complete. Ranking: {final_ranking}")
        return final_ranking


# --- Main execution block (from baseline) ---

if __name__ == "__main__":
    # Set the dataset ("amazon", "goodreads", or "yelp")
    task_set = "amazon" 
    
    # Initialize Simulator
    # Make sure to set your data_dir
    simulator = Simulator(data_dir="your_data_dir_path", device="auto", cache=True)

    # Load scenarios
    simulator.set_task_and_groundtruth(
        task_dir=f"./track2/{task_set}/tasks", 
        groundtruth_dir=f"./track2/{task_set}/groundtruth"
    )

    # Set the new custom agent
    simulator.set_agent(APIBasedRefiningAgent)

    # Set LLM client (e.g., InfinigenceLLM)
    # Make sure to set your api_key
    simulator.set_llm(InfinigenceLLM(api_key="your_api_key_here"))

    # Run evaluation
    agent_outputs = simulator.run_simulation(
        number_of_tasks=None, 
        enable_threading=True, 
        max_workers=10
    )

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    
    # Save results
    output_filename = f'./evaluation_results_track2_{task_set}_APIAgent.json'
    with open(output_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"Evaluation complete. Results saved to {output_filename}")
    print(f"The evaluation_results is: {evaluation_results}")