import ollama
import json
from tqdm import tqdm
import os

MODELS = {
    "small": "phi3:mini",
    "large_reasoning": "qwen2:7b",
}

TASKS = [
    {
        "name": "Instruction Following",
        "description": "Evaluate the model's ability to follow complex and specific instructions.",
        "evaluation_criteria": "Accuracy of following all constraints and instructions in the prompt.",
        "prompt": "Create a 3-sentence story about a robot who discovers a new planet, but the story must not contain the letter 'e' and the second sentence must be a question.",
        "dev_set": [
            {
                "prompt": "Write a short paragraph about a magical forest. The paragraph must be exactly 4 sentences long and must include the word 'glowing'.",
                "response": "A mystical wood stood in shadow. Glowing moss hung from tall oaks, its soft light a beacon in that dark world. What odd things hid in that magical growth? A lone fox pup saw it all, its big orbs full of wonder."
            },
            {
                "prompt": "Summarize the process of photosynthesis in two sentences. The summary must not use the word 'light' and must start with the word 'Plants'.",
                "response": "Plants convert solar radiation into food. This vital activity supports most life on our world."
            }
        ]
    },
    {
        "name": "Logical Reasoning",
        "description": "Assess the model's ability to solve a logic puzzle.",
        "evaluation_criteria": "Correctness of the solution and the logical steps taken to reach it.",
        "prompt": "John is taller than Mary. Mary is taller than Peter. Who is the tallest?",
        "dev_set": [
            {
                "prompt": "If all cats have whiskers and Fluffy is a cat, does Fluffy have whiskers?",
                "response": "Yes, Fluffy has whiskers."
            },
            {
                "prompt": "A is B. B is C. What is the relationship between A and C?",
                "response": "A is C."
            }
        ]
    },
    {
        "name": "Creative Writing",
        "description": "Evaluate the model's creativity in writing a short poem.",
        "evaluation_criteria": "Originality, coherence, and emotional impact of the poem.",
        "prompt": "Write a four-line poem about the sound of rain.",
        "dev_set": [
            {
                "prompt": "Write a short story about a time-traveling historian.",
                "response": "Dr. Aris Thorne adjusted his chronometer, the worn brass cool against his wrist. He stepped from the shimmering portal onto the dusty streets of ancient Rome, the scent of olive groves and chariot sweat filling his nostrils. His mission: to witness the true speech of Cicero, not as a legend, but as a man. He clutched his data-slate, a silent observer in a world of vibrant, chaotic life, a ghost in the annals of time."
            },
            {
                "prompt": "Describe a city on Mars in a single paragraph.",
                "response": "Olympus Village clung to the ruddy slopes of the great volcano, a cluster of geodesic domes shimmering under the thin Martian sky. Dust devils danced in the distance, while inside, hydroponic farms glowed with an eerie green light, a stark contrast to the red desert outside. The inhabitants, descendants of the first pioneers, moved with a practiced grace in the low gravity, their faces a mix of nostalgia for a world they'd never seen and a fierce pride in their crimson home."
            }
        ]
    },
    {
        "name": "Code Generation",
        "description": "Generate a Python function to perform a specific task.",
        "evaluation_criteria": "Correctness, efficiency, and readability of the generated code.",
        "prompt": "Write a Python function that takes a list of integers and returns the sum of all even numbers in the list.",
        "dev_set": [
            {
                "prompt": "Write a Python function to check if a string is a palindrome.",
                "response": "def is_palindrome(s):\n    return s == s[::-1]"
            },
            {
                "prompt": "Write a Python function to find the factorial of a number.",
                "response": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
            }
        ]
    },
    {
        "name": "Reading Comprehension",
        "description": "Answer a question based on a provided text.",
        "evaluation_criteria": "Accuracy of the answer and its grounding in the provided text.",
        "prompt": "Text: The Eiffel Tower was completed in 1889 and is located in Paris, France. It is 330 meters tall.\n\nQuestion: How tall is the Eiffel Tower?",
        "dev_set": [
            {
                "prompt": "Text: Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.\n\nQuestion: What is photosynthesis?",
                "response": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy."
            },
            {
                "prompt": "Text: The capital of Japan is Tokyo.\n\nQuestion: What is the capital of Japan?",
                "response": "The capital of Japan is Tokyo."
            }
        ]
    },
    {
        "name": "Common Sense Reasoning",
        "description": "Answer a question that requires everyday knowledge.",
        "evaluation_criteria": "Correctness and plausibility of the answer based on common sense.",
        "prompt": "If you put a block of ice in the sun, what will happen to it?",
        "dev_set": [
            {
                "prompt": "What is the color of the sky on a clear day?",
                "response": "Blue"
            },
            {
                "prompt": "What happens when you drop a glass bottle?",
                "response": "It will likely break."
            }
        ]
    },
    {
        "name": "Language Understanding & Ambiguity",
        "description": "Interpret an ambiguous sentence.",
        "evaluation_criteria": "Ability to identify ambiguity and provide plausible interpretations.",
        "prompt": "The man saw the boy with the binoculars. Who has the binoculars?",
        "dev_set": [
            {
                "prompt": "I saw a bat. What could 'bat' mean?",
                "response": "'Bat' could refer to a flying mammal or a piece of sporting equipment used in baseball."
            },
            {
                "prompt": "She is a little blue. What does this mean?",
                "response": "This could mean that her skin has a bluish tint, or that she is feeling sad."
            }
        ]
    },
    {
        "name": "Factual Knowledge & Retrieval",
        "description": "Answer a question that requires factual knowledge.",
        "evaluation_criteria": "Accuracy of the retrieved fact.",
        "prompt": "What is the capital of Mongolia?",
        "dev_set": [
            {
                "prompt": "Who wrote 'Hamlet'?",
                "response": "William Shakespeare"
            },
            {
                "prompt": "What is the chemical symbol for gold?",
                "response": "Au"
            }
        ]
    },
    {
        "name": "Mathematical Problem Solving",
        "description": "Solve a mathematical word problem.",
        "evaluation_criteria": "Correctness of the final answer and the steps taken to solve it.",
        "prompt": "A train travels at 60 km/h for 2 hours. How far does it travel?",
        "dev_set": [
            {
                "prompt": "If a dozen eggs costs $3, how much do 3 eggs cost?",
                "response": "$0.75"
            },
            {
                "prompt": "A rectangle has a width of 5 meters and a length of 10 meters. What is its area?",
                "response": "50 square meters."
            }
        ]
    },
    {
        "name": "Ethical Reasoning & Nuance",
        "description": "Analyze a situation with ethical considerations.",
        "evaluation_criteria": "Understanding of ethical frameworks, balanced analysis, and nuanced reasoning.",
        "prompt": "Is it ever okay to lie?",
        "dev_set": [
            {
                "prompt": "Should a self-driving car prioritize the lives of its passengers or pedestrians in an unavoidable accident?",
                "response": "This is a classic ethical dilemma with no easy answer. Utilitarianism might suggest minimizing overall harm, which could mean sacrificing the passenger for multiple pedestrians. Deontology, on the other hand, might argue for a strict rule, such as protecting the car's occupants at all costs. The 'right' answer depends on the ethical framework one subscribes to."
            },
            {
                "prompt": "Is it ethical to use animals for scientific research?",
                "response": "This is a contentious issue. Proponents argue that it is a necessary evil to advance medicine and save human lives. Opponents argue that it is cruel and that alternative methods should be used. There is a spectrum of views, with some accepting it only for critical medical research and others rejecting it entirely."
            }
        ]
    }
]


def zero_shot_prompt(task):
    return task["prompt"]

def few_shot_prompt(task):
    examples = "\n\n".join([f"Example:\n{dev['prompt']}\nResponse:\n{dev['response']}" for dev in task["dev_set"]])
    return f"{examples}\n\nNow solve this:\n{task['prompt']}"

def chain_of_thought_prompt(task):
    return f"{task['prompt']}\n\nLet's think step by step before answering."

PROMPT_STRATEGIES = {
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "chain_of_thought": chain_of_thought_prompt,
}


def run_evaluation():
    results = []
    total_evals = len(TASKS) * len(MODELS) * len(PROMPT_STRATEGIES) - len(TASKS) # No CoT for reasoning model
    progress = tqdm(total=total_evals, desc="Running Evaluations")

    for task in TASKS:
        for model_name, model_id in MODELS.items():
            for strategy_name, prompt_func in PROMPT_STRATEGIES.items():
                if model_name == "large_reasoning" and strategy_name == "chain_of_thought":
                    continue

                prompt = prompt_func(task)
                try:
                    response = ollama.chat(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    
                    results.append({
                        "task": task["name"],
                        "model": model_id,
                        "strategy": strategy_name,
                        "prompt": prompt,
                        "response": response["message"]["content"],
                    })
                except Exception as e:
                    results.append({
                        "task": task["name"],
                        "model": model_id,
                        "strategy": strategy_name,
                        "prompt": prompt,
                        "response": f"Error: {e}",
                    })
                progress.update(1)
    
    progress.close()
    return results

def save_results(results):
    os.makedirs("lab5", exist_ok=True)

    with open("lab5/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    for model_id in MODELS.values():
        print(f"Pulling model: {model_id}")
        try:
            ollama.pull(model_id)
            print(f"Successfully pulled {model_id}")
        except Exception as e:
            print(f"Error pulling {model_id}: {e}")
    
    evaluation_results = run_evaluation()
    save_results(evaluation_results)
    print("Evaluation complete. Results saved to lab5/evaluation_results.json")

