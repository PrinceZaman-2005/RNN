# Model Notes

## 1. Overview

This model implements a basic **Recurrent Neural Network (RNN)** using PyTorch.

Its purpose is to process a sequence of numerical values and predict the next value using information encoded in the final recurrent output.

For example:

```text
Input:  [2, 4, 6, 8]
Target: 10
```

The model learns this relationship through training examples rather than relying on a predefined mathematical rule.

## 2. Model Architecture

The model contains two primary components:

1. **Recurrent layer:** Processes the input sequence and updates its hidden state at every timestep.
2. **Linear layer:** Converts the final recurrent output into the predicted value.

```text
        Input Sequence
        [2, 4, 6, 8]
               |
               v
        ┌─────────────┐
        │  RNN Layer  │
        │             │
        │ h₁ → h₂     │
        │      → h₃   │
        │      → h₄   │
        └─────────────┘
               |
               v
       Final RNN Output
               |
               v
        ┌─────────────┐
        │ Linear Layer│
        └─────────────┘
               |
               v
       Predicted Next Value
```

The diagram represents the recurrent layer unrolled across the sequence. The same RNN parameters are shared across all timesteps.

## 3. Implementation

The model is defined in `core/rnn.py`.

## 4. Understanding the Parameters

| Parameter     | Default | Meaning                                                          |
| ------------- | ------: | ---------------------------------------------------------------- |
| `input_size`  |       1 | Number of features at each timestep                              |
| `hidden_size` |      16 | Number of features in the recurrent hidden state                 |
| `num_layers`  |       1 | Number of stacked recurrent layers                               |
| `output_size` |       1 | Number of predicted values                                       |
| `batch_first` |  `True` | Places the batch dimension first in the input and output tensors |

These parameters allow the model to be adapted to different sequential prediction tasks.

For example, `input_size=3` could represent three measurements at each timestep, while `output_size=2` could produce two predicted values.

## 5. How the RNN Processes Information

At each timestep, the RNN combines the current input with the previous hidden state.

Conceptually, its computation is:

$$
h_t = \tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{t-1} + b_{hh})
$$

Where:

* \(x_t\): Input at the current timestep.
* \(h_{t-1}\): Hidden state from the previous timestep.
* \(h_t\): Updated hidden state.
* \(W_{ih}\): Input-to-hidden weights.
* \(W_{hh}\): Hidden-to-hidden weights.
* \(b_{ih}, b_{hh}\): Bias terms.
* \(\tanh\): Nonlinear activation function.

The initial hidden state is zero by default when one is not explicitly supplied.

The updated hidden state carries information forward through the sequence. However, it is a learned representation, not a perfect record of every previous input.

## 6. Forward Pass

Consider the input sequence:

```python
x = torch.tensor([
    [[2.0], [4.0], [6.0], [8.0]]
])
```

Its shape is:

```text
(batch_size, sequence_length, input_size)

(1, 4, 1)
```

### Step 1: Recurrent processing

```python
output, hidden = self.rnn(x)
```

The RNN processes all four timesteps.

For a single recurrent layer, the shapes are:

```text
output: (1, 4, 16)
hidden: (1, 1, 16)
```

* `output` contains the hidden output for every timestep.
* `hidden` contains the final hidden state for each recurrent layer.

### Step 2: Select the final output

```python
last_output = output[:, -1, :]
```

This selects the output corresponding to the last timestep.

Shape:

```text
(1, 16)
```

The model uses this representation to make its next-value prediction.

### Step 3: Generate the prediction

```python
prediction = self.fc(last_output)
```

The linear layer maps the 16-dimensional recurrent output to one numerical value.

Output shape:

```text
(1, 1)
```

The predicted value is not guaranteed to be `10`. It depends on the learned parameters and training process.

## 7. How Does the Model Learn?

The model can be trained using Mean Squared Error (MSE) for numerical prediction.

The loss is:

$$
L = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

Where:

* \(y_i\): Actual target.
* \(\hat{y}_i\): Predicted value.
* \(N\): Number of examples.

During training:

1. The model processes the input sequence.
2. The linear layer produces a prediction.
3. The loss measures the difference between the prediction and the target.
4. Backpropagation calculates gradients through the output layer and recurrent computations.
5. The optimizer updates the parameters.

Because the recurrent computation is unrolled across timesteps, this training process is known as **Backpropagation Through Time (BPTT)**.

## 8. Limitations

This is a basic RNN, intentionally kept simple for educational purposes.

### Sequential computation

Each timestep depends on the preceding hidden state. Therefore, recurrent computation within a sequence cannot generally be parallelized in the same way as independent feed-forward operations.

### Vanishing and exploding gradients

During BPTT, gradients may become very small or very large as they propagate through many timesteps. This can make learning long-term dependencies difficult.

### Limited long-term memory

The hidden state has a fixed dimensionality. Although it can encode information from earlier inputs, it may fail to preserve information needed much later in a sequence.

### Learning depends on the dataset

A model trained on simple numerical sequences may learn those patterns without generalizing to more complex sequences.

## 9. Learning Objective

This model demonstrates how recurrent computation allows a neural network to process ordered data while carrying a hidden representation from one timestep to the next.

The central concepts are:

* Recurrent connections.
* Hidden-state updates.
* Shared parameters across timesteps.
* Sequence-to-one prediction.
* Backpropagation Through Time.
* The challenges of learning long-term dependencies.

