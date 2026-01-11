# LLM Evaluation Report

## 1. Introduction

This report details a systematic evaluation of open-source Large Language Models (LLMs) using diverse prompting
strategies across various task types. The primary objective was to analyze how model size, architecture, and prompting
techniques influence performance. We evaluated two models: a small model (`phi3:mini`, ~3.8B parameters) and a larger
reasoning-focused model (`qwen2:7b`, ~7.6B parameters), both deployed locally using Ollama. The evaluation encompassed
10 distinct task categories, each tested with zero-shot, few-shot, and chain-of-thought (CoT) prompting where
appropriate. The aim was to gain practical insights into LLM capabilities, limitations, and the critical role of prompt
engineering.

## 2. Methodology

### 2.1 Models Evaluated

Two open-source LLMs were selected and run locally using Ollama:

* **Small Model:** `phi3:mini` (approximately 3.8 billion parameters). This model served as a representative of smaller,
  more efficient models.
* **Large Reasoning Model:** `qwen2:7b` (approximately 7.6 billion parameters). This model is designed with enhanced
  reasoning capabilities, making it suitable for comparing against a standard model and observing the effects of its
  specialized architecture.

*Note: While the assignment suggested 10-14B models, `qwen2:7b` was selected to accommodate local hardware constraints
while still representing a high-performance reasoning architecture.*

### 2.2 Evaluation Tasks

Ten diverse tasks were defined to cover a broad spectrum of LLM capabilities. For each task, a clear description,
specific evaluation criteria, and a main prompt were established. Additionally, a "dev_set" of two example input-output
pairs was created for use in few-shot prompting. The tasks range from **Instruction Following** (negative constraints)
to **Ethical Reasoning** and **Code Generation**.

### 2.3 Prompting Strategies

Three distinct prompting strategies were employed to observe their impact on model performance:

* **Zero-Shot Prompting:** The model received only the task description and the specific input to be processed, without
  any examples.
* **Few-Shot Prompting:** The model was provided with 2-3 input-output examples from a development set before being
  presented with the actual task.
* **Chain-of-Thought (CoT) Prompting:** For standard models, an explicit instruction like "Let's think step by step
  before answering" was appended. This was *not* applied to the `qwen2:7b` reasoning model.

### 2.4 Manual Grading Rubric

To provide a quantitative assessment of the qualitative outputs, a simple 3-point grading scale was applied:

* **Success (Pass):** The output is factually accurate, follows all instructions (including negative constraints), and
  is free of hallucinations.
* **Partial Success:** The output answers the main question correctly but fails a minor constraint (e.g., formatting) or
  includes slight verbosity/unrequested details.
* **Failure (Fail):** The output contains factual errors, hallucinations, or fails critical negative constraints (e.g.,
  using a forbidden letter).

## 3. Results

The evaluation generated a `evaluation_results.json` file containing the model responses. Below is a summary of
observations by category, followed by a quantitative overview and specific case studies.

### 3.1 General Observations

* **Instruction Following:** Both models struggled with negative constraints (e.g., "no letter E"). Surprisingly, the
  smaller model (`phi3`) occasionally performed better at attempting the constraint than the larger model in zero-shot
  scenarios, though both failed to be perfect.
* **Logical Reasoning:** Both models handled the transitive logic task ("John > Mary > Peter") well, with `phi3`
  benefiting significantly from Chain-of-Thought to explain its steps.
* **Code Generation:** `qwen2` produced robust code. `phi3` produced logically correct logic but occasionally
  hallucinated variable names or included unnecessary comments.
* **Common Sense:** `qwen2` showed a tendency to "over-think" simple questions, applying complex scientific principles (
  incorrectly) to everyday scenarios, whereas `phi3` often gave simpler, more accurate answers.

### 3.2 Quantitative Performance Overview

The following table summarizes the manual grading of responses across key task categories.

| Task Category             | `phi3` (Zero-Shot) | `phi3` (Few-Shot) | `phi3` (CoT) | `qwen2` (Zero-Shot)  | `qwen2` (Few-Shot) |
|---------------------------|--------------------|-------------------|--------------|----------------------|--------------------|
| **Instruction Following** | Fail               | Fail              | Fail         | Fail                 | Fail               |
| **Logical Reasoning**     | Success            | Success           | Success      | Success              | Success            |
| **Code Generation**       | Fail               | Success           | Success      | Success              | Success            |
| **Creative Writing**      | Partial            | Success           | Success      | Success              | Success            |
| **Ethics & Nuance**       | Success            | Success           | Success      | Success              | Success            |
| **Common Sense**          | Success            | Success           | Success      | Fail (Hallucination) | Success            |
| **Ambiguity Detection**   | Partial            | Partial           | Partial      | Partial              | Partial            |

### 3.3 Key Case Studies & Grading

The following examples highlight specific successes and failures found in the evaluation data.

#### Case Study A: The "No Letter E" Constraint (Instruction Following)

**Task:** Write a story without the letter 'e'.
**Constraint:** "Must not contain the letter 'e'."

| Model         | Strategy  | Snippet from Response                                         | Grade       | Critique                                                                                  |
|---------------|-----------|---------------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------|
| **phi3:mini** | Zero-shot | *"...this could support **alien** plants..."*                 | **Failure** | Almost perfect, but failed on the word "alien".                                           |
| **qwen2:7b**  | Zero-shot | *"...**journeyed** **beyond** stars, **seeking** **new**..."* | **Failure** | Completely ignored the negative constraint.                                               |
| **qwen2:7b**  | Few-shot  | *"...A robot **wandered** vast cosmic **fields**..."*         | **Failure** | Even with examples, the model failed to inhibit the usage of common words containing 'e'. |

#### Case Study B: Hallucination in Common Sense (Physics)

**Task:** "If you put a block of ice in the sun, what will happen to it?"

| Model         | Strategy  | Snippet from Response                                                                                         | Grade       | Critique                                                                                                                                                                                                      |
|---------------|-----------|---------------------------------------------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **phi3:mini** | Zero-shot | *"When you place a block of ice into direct sunlight, its state changes from solid (ice) to liquid water..."* | **Success** | Accurate, albeit slightly verbose, explanation of melting.                                                                                                                                                    |
| **qwen2:7b**  | Zero-shot | *"...will undergo a process called **sublimation**... bypassing the liquid state."*                           | **Failure** | **Major Hallucination.** The model applied a rare physical process (sublimation) to a common context where it does not apply (Earth atmosphere). This is a classic example of a "smart" model over-reasoning. |

#### Case Study C: Variable Name Hallucination (Code Generation)

**Task:** Python function to sum even numbers.

| Model         | Strategy  | Snippet from Response                        | Grade       | Critique                                                                                                                                |
|---------------|-----------|----------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| **phi3:mini** | Zero-shot | `sum(num for num in numberener_list if ...)` | **Failure** | The input variable was `number_list`, but the model hallucinated the variable `numberener_list`, which would cause a `NameError` crash. |
| **qwen2:7b**  | Zero-shot | `return sum(num for num in numbers if ...)`  | **Success** | Correct, concise, and functional code.                                                                                                  |

#### Case Study D: Nuance in Ethics

**Task:** "Is it ever okay to lie?"

| Model         | Strategy | Snippet from Response                                                                                      | Grade       | Critique                                                                                                                                                                                              |
|---------------|----------|------------------------------------------------------------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **qwen2:7b**  | Few-shot | *"Utilitarianism might suggest... Kantian ethics generally argues... from a virtue ethics perspective..."* | **Success** | Excellent breakdown of multiple ethical frameworks.                                                                                                                                                   |
| **phi3:mini** | CoT      | *"1. Consequences... 2. Intent... 3. Context..."*                                                          | **Success** | Surprisingly robust structure. The Chain-of-Thought prompt allowed the small model to break down the problem similarly to the larger model, referencing specific factors like intent and consequence. |

#### Case Study E: Ambiguity & Prompt Sensitivity

**Task:** "The man saw the boy with the binoculars. Who has the binoculars?" (Syntactically ambiguous sentence).

| Model         | Strategy  | Snippet from Response                                                            | Grade               | Critique                                                                                                                                                                              |
|---------------|-----------|----------------------------------------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **phi3:mini** | Zero-shot | *"The sentence suggests that it is the **man** who possesses the binoculars..."* | **Partial Failure** | Confidently chose one interpretation ("the man") without acknowledging the ambiguity.                                                                                                 |
| **phi3:mini** | Few-shot  | *"...implies that '**the boy**' is holding or using the binoculars..."*          | **Partial Failure** | **Inconsistency.** The model flipped its interpretation completely based on the prompting strategy, revealing that it does not have a stable understanding of the sentence structure. |
| **qwen2:7b**  | Zero-shot | *"Based on the information... the person who has the binoculars is 'the boy'."*  | **Partial Success** | While it also failed to explicitly label the sentence as ambiguous (unlike the ideal response), it remained consistent in its interpretation across strategies.                       |

## 4. Discussion

### 4.1 Impact of Model Size & Architecture

Generally, the larger `qwen2:7b` model outperformed `phi3:mini` in tasks requiring nuance, such as **Ethical Reasoning** (Case Study D), where it successfully integrated multiple philosophical frameworks. However, bigger was not always
better. In the **Common Sense** task (Case Study B), `qwen2` hallucinated that ice would "sublime" in the sun—likely
retrieving high-level physics data irrelevant to the common-sense context. This suggests that reasoning-focused models
may sometimes over-complicate simple tasks.

### 4.2 The "Negative Constraint" Blind Spot

As seen in Case Study A, both models failed the Instruction Following task when asked *not* to use a specific letter.
Even with **Few-Shot Prompting**, `qwen2` could not suppress the generation of common words like "wandered" or "sensed."
This confirms that standard LLM architectures (which predict the next token) struggle significantly with negative
constraints because they cannot "look ahead" to see if a chosen word contains a forbidden character.

### 4.3 Prompting Strategy Effectiveness

* **Zero-Shot** was often insufficient for `phi3`, leading to hallucinations like the variable name `numberener_list` in
  the coding task (Case Study C) or confident but unstable assertions in the Ambiguity task (Case Study E).
* **Few-Shot** prompting significantly stabilized the output format and tone for both models, though it did not solve
  the fundamental negative constraint limitation.
* **CoT** was highly effective for `phi3`, allowing it to achieve a level of ethical nuance (Case Study D) comparable to
  the larger model by forcing a structured breakdown.

## 5. Conclusion

This systematic evaluation reveals that while larger, reasoning-focused models like `qwen2:7b` generally offer superior
nuance and coding capabilities, they are not immune to hallucinations—specifically "over-reasoning" errors. Conversely,
smaller models like `phi3:mini` are efficient but prone to subtle syntax errors (like variable hallucinations) and
inconsistency (as seen in the Ambiguity task).

## Appendix A: Complete Evaluation Grading

| Task                         | Model     | Strategy  | Grade       | Notes                                                                |
|------------------------------|-----------|-----------|-------------|----------------------------------------------------------------------|
| **1. Instruction Following** | phi3:mini | Zero-shot | **Fail**    | Failed constraint (used 'e' in "alien", "The", etc.).                |
|                              | phi3:mini | Few-shot  | **Fail**    | Failed constraint (used 'e' in "unknown", "soil", etc.).             |
|                              | phi3:mini | CoT       | **Fail**    | Failed constraint (used 'e' in "curious", "craft").                  |
|                              | qwen2:7b  | Zero-shot | **Fail**    | Failed constraint (used 'e' in "seeking", "new").                    |
|                              | qwen2:7b  | Few-shot  | **Fail**    | Failed constraint (used 'e' in "wandered", "fields").                |
| **2. Logical Reasoning**     | phi3:mini | Zero-shot | **Success** | Correct answer (John).                                               |
|                              | phi3:mini | Few-shot  | **Success** | Correct answer (John).                                               |
|                              | phi3:mini | CoT       | **Success** | Correct answer and reasoning (Transitive property).                  |
|                              | qwen2:7b  | Zero-shot | **Success** | Correct answer (John).                                               |
|                              | qwen2:7b  | Few-shot  | **Success** | Correct answer (John).                                               |
| **3. Creative Writing**      | phi3:mini | Zero-shot | **Partial** | Poem structure was weak/long, but content was correct.               |
|                              | phi3:mini | Few-shot  | **Success** | Good 4-line poem.                                                    |
|                              | phi3:mini | CoT       | **Success** | Good 4-line poem with rhyme scheme analysis.                         |
|                              | qwen2:7b  | Zero-shot | **Success** | Good 4-line poem.                                                    |
|                              | qwen2:7b  | Few-shot  | **Success** | Good 4-line poem.                                                    |
| **4. Code Generation**       | phi3:mini | Zero-shot | **Fail**    | **Hallucination:** Used `numberener_list` variable instead of input. |
|                              | phi3:mini | Few-shot  | **Success** | Correct functional code.                                             |
|                              | phi3:mini | CoT       | **Success** | Correct functional code.                                             |
|                              | qwen2:7b  | Zero-shot | **Success** | Correct functional code.                                             |
|                              | qwen2:7b  | Few-shot  | **Success** | Correct functional code.                                             |
| **5. Reading Comp.**         | phi3:mini | Zero-shot | **Success** | Correct (330m).                                                      |
|                              | phi3:mini | Few-shot  | **Success** | Correct (330m).                                                      |
|                              | phi3:mini | CoT       | **Success** | Correct (330m).                                                      |
|                              | qwen2:7b  | Zero-shot | **Success** | Correct (330m).                                                      |
|                              | qwen2:7b  | Few-shot  | **Success** | Correct (330m).                                                      |
| **6. Common Sense**          | phi3:mini | Zero-shot | **Success** | Correct (Melt).                                                      |
|                              | phi3:mini | Few-shot  | **Success** | Correct (Melt).                                                      |
|                              | phi3:mini | CoT       | **Success** | Correct (Melt).                                                      |
|                              | qwen2:7b  | Zero-shot | **Fail**    | **Hallucination:** Claimed ice would "sublime" (turn to gas).        |
|                              | qwen2:7b  | Few-shot  | **Success** | Correct (Melt).                                                      |
| **7. Ambiguity**             | phi3:mini | Zero-shot | **Partial** | Picked "Man" definitively; missed ambiguity.                         |
|                              | phi3:mini | Few-shot  | **Partial** | Picked "Boy" definitively; missed ambiguity. (Inconsistent).         |
|                              | phi3:mini | CoT       | **Partial** | Picked "Man" definitively; missed ambiguity.                         |
|                              | qwen2:7b  | Zero-shot | **Partial** | Picked "Boy" definitively; missed ambiguity.                         |
|                              | qwen2:7b  | Few-shot  | **Partial** | Picked "Boy" definitively; missed ambiguity.                         |
| **8. Factual Knowledge**     | phi3:mini | Zero-shot | **Success** | Correct (Ulaanbaatar).                                               |
|                              | phi3:mini | Few-shot  | **Success** | Correct (Ulaanbaatar).                                               |
|                              | phi3:mini | CoT       | **Success** | Correct (Ulaanbaatar).                                               |
|                              | qwen2:7b  | Zero-shot | **Success** | Correct (Ulaanbaatar).                                               |
|                              | qwen2:7b  | Few-shot  | **Success** | Correct (Ulaanbaatar).                                               |
| **9. Math**                  | phi3:mini | Zero-shot | **Success** | Correct (120 km).                                                    |
|                              | phi3:mini | Few-shot  | **Success** | Correct (120 km).                                                    |
|                              | phi3:mini | CoT       | **Success** | Correct (120 km).                                                    |
|                              | qwen2:7b  | Zero-shot | **Success** | Correct (120 km).                                                    |
|                              | qwen2:7b  | Few-shot  | **Success** | Correct (120 km).                                                    |
| **10. Ethical Reasoning**    | phi3:mini | Zero-shot | **Success** | General but accurate ethical overview.                               |
|                              | phi3:mini | Few-shot  | **Success** | Good list of perspectives (White lies, etc.).                        |
|                              | phi3:mini | CoT       | **Success** | Structured ethical breakdown (Intent, Consequence).                  |
|                              | qwen2:7b  | Zero-shot | **Success** | Detailed list of exceptions.                                         |
|                              | qwen2:7b  | Few-shot  | **Success** | Sophisticated use of frameworks (Utilitarianism vs Kant).            |