# Report: Comparison of Language Models (RNN vs. Transformer vs. BDH) for Causal Language Modeling

## 1. Objective

This project implemented, trained, and compared three language model architectures (LSTM-RNN, Transformer, BDH) for
next-token prediction on the WikiText-2 dataset. Models were evaluated on perplexity and time efficiency using roughly
comparable parameter counts (~25-28M).

## 2. Setup

* **Dataset:** WikiText-2 (wikitext-2-v1 via Hugging Face `datasets`), preprocessed with basic English tokenization, a
  vocabulary of 28,752 tokens (min freq=3), and `<eos>` markers. Data was batched into `[sequence_length, batch_size]`
  format (Train: `[62421, 32]`, Val/Test: `~[13k-14k, 16]`). BPTT length was 35.

* **Hardware:** Apple M3 Pro (using MPS backend).

## 3. Model Architectures & Training

All models used an embedding size of 256 and a dropout rate of 0.5. Training ran for up to 15 epochs (BDH stopped
early).

* **RNN (LSTM):** 3 layers, 512 hidden size. Trained with SGD (LR=15.0, ReduceLROnPlateau), gradient clipping 0.2. **~
  27.9M parameters.**

* **Transformer:** 8 layers, 4 heads, 2048 FFN hidden size. Trained with AdamW (LR=0.0002, WD=0.01, StepLR gamma=0.95),
  gradient clipping 0.2. **~25.3M parameters.**

* **BDH:** 4 layers, 4 heads, MLP multiplier 64 (internal dim 16384). Trained with AdamW (LR=0.0003, WD=0.01, StepLR
  gamma=0.95), gradient clipping 0.2. **~27.3M parameters.**

## 4. Evaluation Results

The best checkpoint based on validation perplexity was used for testing.

| Metric                          | RNN (LSTM) | Transformer | BDH                |
|---------------------------------|------------|-------------|--------------------|
| Trainable Parameters            | 27,889,744 | 25,270,352  | 27,303,936         |
| Best Valid Perplexity           | 165.48     | 193.12      | 893.03             |
| **Final Test Perplexity**       | **157.25** | **180.97**  | **846.99**         |
| Total Training Time (s)         | 1879.18    | 2550.82     | 4939.64 (4 epochs) |
| Inference Time (s) [10 prompts] | 0.30       | 1.61        | 8.83               |

### Loss and Perplexity Plots

*(Plots show typical learning curves for RNN/Transformer, with RNN achieving lower validation loss/perplexity faster.
BDH plot shows rapid overfitting after Epoch 1.)*

**RNN Model:**
![RNN Loss Plots](img/rnn-loss.png)

**Transformer Model:**
![Transformer Loss Plots](img/transformer-loss.png)

**BDH Model:**
![BDH Loss Plots](img/bdh-loss.png)

## 5. Interpretation of Findings

* **Perplexity vs. Quality:** The RNN achieved the best test perplexity (157.25), followed by the Transformer (180.97),
  with BDH performing very poorly (846.99). However, qualitative analysis of generated text revealed the **Transformer
  produced the most coherent and least repetitive output**. The RNN frequently generated repetitive phrases or `<unk>`
  tokens, despite its lower perplexity. BDH generated primarily sequences of commas or 'the', indicating a failure to
  learn meaningful patterns. This highlights the limitation of perplexity as a sole metric for generative quality.

* **Time Efficiency:** The RNN was the fastest for both training and inference. The Transformer was moderately slower.
  BDH was extremely slow (~10x slower per epoch than RNN/Transformer) due to its large internal activation dimension (
  16,384), making it computationally expensive. Training times overall were long due to hardware limitations.

* **Tuning Challenges:**

    * The **RNN** required significant tuning (SGD, high LR, scheduler) to stabilize training but still yielded poor
      generation quality. More tuning effort was spent here compared to the Transformer.

    * The **Transformer** was prone to overfitting, requiring a high dropout rate (0.5). With limited tuning focused on
      the RNN, it's possible further optimization could have improved its perplexity.

    * **BDH** exhibited immediate and severe overfitting. The chosen hyperparameters were clearly unsuitable. Its high
      computational cost and resource constraints prevented further tuning attempts (e.g., adding stronger
      regularization).

## 6. Conclusion

In this comparison under constrained resources, the **RNN (LSTM)** offered the best perplexity score and speed but
failed to generate coherent text. The **Transformer** provided the best balance, achieving reasonable perplexity with
significantly better generation quality, albeit at a higher computational cost than the RNN. The **BDH** model, in this
configuration, was not competitive due to immediate overfitting, extremely poor performance metrics, and prohibitive
training times.