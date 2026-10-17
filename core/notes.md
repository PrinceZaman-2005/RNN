# RNN — IMDb Sentiment Classification

## 1. Dataset

This project uses the **IMDb Large Movie Review Dataset** to classify movie reviews as positive or negative.

**Dataset:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (Hugging Face)

| Property                |                      Value |
| ----------------------- | -------------------------: |
| Total reviews           |                     50,000 |
| Training reviews        |                     25,000 |
| Testing reviews         |                     25,000 |
| Classes                 | Positive (1), Negative (0) |
| Maximum vocabulary size |                     10,000 |
| Maximum review length   |                 200 tokens |

### Data Preprocessing

1. Convert reviews to lowercase.
2. Tokenize the text into words.
3. Build a vocabulary using training reviews only.
4. Convert words into integer token IDs.
5. Replace unknown words with `<UNK>`.
6. Pad shorter reviews and truncate longer ones to 200 tokens.

**Special tokens:**

* `<PAD>` → `0`
* `<UNK>` → `1`

**Input shape:** `(batch_size, sequence_length)`

For a batch of 32 reviews:

```text
(32, 200)
```

---

## 2. Model Architecture

The model consists of three main components: an **Embedding Layer, a Vanilla RNN, and a Linear Output Layer**.

```text
       Movie Review
             |
             v
          Token IDs
             |
             v
       Embedding Layer
             |
             v
          Vanilla RNN
             |
             v
      Final Hidden State
             |
             v
         Linear Layer
             |
             v
       Sentiment Logit
             |
             v
     Positive / Negative
```

### A. Embedding Layer

The embedding layer converts token IDs into learnable, dense vector representations.

```python
nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=64,
    padding_idx=0
)
```

With a vocabulary of 10,000 tokens and an embedding dimension of 64, the embedding matrix has the shape `(10000, 64)`.

For a batch of 32 reviews:

```text
Input:     (32, 200)
Embedded:  (32, 200, 64)
```

The embeddings are learned during training rather than loaded from a pretrained model.

### B. Recurrent Neural Network

The RNN processes the embedded sequence one timestep at a time. At each timestep, it updates its hidden state using the current input and the previous hidden state.

**RNN configuration:**

| Parameter        |       Value |
| ---------------- | ----------: |
| Input size       |          64 |
| Hidden size      |          64 |
| Number of layers |           1 |
| Activation       |        Tanh |
| Batch format     | Batch first |

The hidden state is calculated as:

$$
h_t = \tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{t-1} + b_{hh})
$$

Where:

* \(x_t\): Input embedding at timestep \(t\).
* \(h_t\): Current hidden state.
* \(h_{t-1}\): Previous hidden state.
* \(W_{ih}\) and \(W_{hh}\): Learnable weight matrices.
* \(b_{ih}\) and \(b_{hh}\): Bias terms.

The hidden state carries information forward through the sequence. The final hidden state is used as the representation for sentiment classification.

### C. Linear Output Layer

The linear layer maps the final hidden representation to a single sentiment logit.

```python
nn.Linear(64, 1)
```

For a batch of 32 reviews:

```text
Final hidden state: (32, 64)
Output logits:      (32,)
```

The output logit can be converted into a probability using the sigmoid function:

$$
p = \frac{1}{1 + e^{-z}}
$$

Where \(z\) is the output logit and \(p\) represents the predicted probability of a positive review.

**Classification rule:**

* Probability ≥ 0.5: Positive
* Probability < 0.5: Negative

---

## 3. How Does the Model Learn?

Since this project performs binary sentiment classification, the model uses **Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)**.

This loss function combines the sigmoid operation and binary cross-entropy into a numerically stable calculation.

For a single example, the loss is:

$$
L = -\left[y\log(p) + (1-y)\log(1-p)\right]
$$

Where:

* \(y\): Actual label (0 or 1).
* \(p\): Predicted probability of the positive class.

### Training Process

1. The model receives a batch of tokenized reviews.
2. The embedding layer converts token IDs into vectors.
3. The RNN processes the sequence and produces a final hidden state.
4. The linear layer generates sentiment logits.
5. The loss function measures the prediction error.
6. Backpropagation calculates gradients through the output layer, RNN, and embedding layer.
7. The optimizer updates the model parameters.

Because the recurrent computation is unfolded across timesteps during gradient calculation, this training process is known as **Backpropagation Through Time (BPTT)**.

---

## 4. Limitations

This is a basic Vanilla RNN, intentionally kept simple for educational purposes.

### Sequential Computation

Each timestep depends on the preceding hidden state. Consequently, recurrent computations within a sequence cannot generally be parallelized across timesteps in the same way as independent feed-forward operations.

### Vanishing and Exploding Gradients

During BPTT, gradients can become very small or very large as they propagate through many timesteps. This makes learning long-term dependencies difficult.

### Limited Long-Term Memory

The hidden state has a fixed dimensionality. Although it can encode information from earlier inputs, it may struggle to preserve relevant information across long sequences.

### Padding and Truncation

Reviews are padded or truncated to a fixed length of 200 tokens. Padding can influence the final hidden state unless handled appropriately, while truncation removes information beyond the selected length.

### Dependence on Training Data

The model learns patterns from the training reviews. Its ability to classify unseen reviews depends on how well those learned patterns generalize beyond the training data.

---

## 5. Learning Objectives

This project demonstrates how recurrent computation enables a neural network to process ordered data while maintaining a hidden representation across timesteps.

The key concepts are:

* **Embeddings:** Learning vector representations of tokens.
* **Recurrent connections:** Passing information between timesteps.
* **Hidden-state updates:** Combining current inputs with previous state information.
* **Shared parameters:** Reusing the same recurrent weights across timesteps.
* **Sequence classification:** Mapping an entire review to a sentiment prediction.
* **Backpropagation Through Time:** Training recurrent computations through gradient propagation.
* **Long-term dependency challenges:** Understanding why basic RNNs can struggle with extended sequences.

The next step is to explore **Long Short-Term Memory (LSTM)** networks, which introduce gating mechanisms to help preserve and control information across longer sequences.
