# 🔄 Recurrent Neural Network (RNN)

A simple Recurrent Neural Network (RNN) built with PyTorch to explore sequential data, hidden states, and recurrent computation.

This project demonstrates how RNNs process sequences one timestep at a time, using information from previous timesteps to influence their current computations.

## ⚡ Introduction

A **Recurrent Neural Network (RNN)** is a neural network architecture designed to process sequential data.

Unlike a conventional feed-forward network, an RNN maintains a hidden state that is updated as it processes each element of a sequence. This state provides a way to carry information from earlier timesteps into later computations.

RNNs can be used for tasks involving time series, text, and other forms of sequential data.

### Architecture

```text
          Input Sequence
       x₁    x₂    x₃    x₄
       |     |     |     |
       v     v     v     v
     ┌────┐ ┌────┐ ┌────┐ ┌────┐
h₀ → │ RNN│→│ RNN│→│ RNN│→│ RNN│
     │Cell│ │Cell│ │Cell│ │Cell│
     └────┘ └────┘ └────┘ └────┘
       |     |     |     |
       v     v     v     v
      h₁    h₂    h₃    h₄
                         |
                         v
                   Output Layer
                         |
                         v
                    Prediction
```

Each RNN cell uses the current input and the previous hidden state to calculate a new hidden state.

The diagram illustrates the recurrent computation unrolled across four timesteps. The same recurrent parameters are shared across these timesteps.

### How Does It Work?

1. **Input sequence:** The model receives an ordered sequence of values.
2. **Initial hidden state:** Processing begins with an initial hidden state, commonly initialized to zeros.
3. **Recurrent computation:** At each timestep, the RNN combines the current input with the previous hidden state.
4. **Hidden-state update:** The network calculates a new hidden state containing information from the current computation.
5. **Output generation:** An output layer uses the relevant hidden state to produce a prediction.
6. **Loss calculation:** The model compares its prediction with the target value.
7. **Learning:** Backpropagation Through Time calculates gradients across the unrolled sequence, allowing the optimizer to update the network's parameters.

The objective is to learn sequential patterns that help the model make predictions from ordered data.

> An RNN can carry information through hidden states, but this does not guarantee that it will preserve every relevant detail of a sequence.

## Why Use PyTorch?

Implementing an RNN requires managing recurrent computations, calculating gradients, and updating model parameters.

PyTorch provides tools that simplify these tasks.

| Task                       | With PyTorch                                        |
| -------------------------- | --------------------------------------------------- |
| Define recurrent layers    | `torch.nn.RNN`                                      |
| Define output layers       | `torch.nn.Linear`                                   |
| Apply activation functions | `torch.nn.Tanh` or other supported activations      |
| Calculate prediction loss  | `torch.nn.MSELoss` or other suitable loss functions |
| Calculate gradients        | Automatic differentiation with `autograd`           |
| Update parameters          | `torch.optim`                                       |
| Process sequential data    | PyTorch tensors                                     |

For example, a recurrent layer can be defined using:

```python
nn.RNN(
    input_size=1,
    hidden_size=16,
    batch_first=True
)
```

Here:

* `input_size=1` means each timestep contains one input feature.
* `hidden_size=16` means the recurrent hidden state has 16 features.
* `batch_first=True` means the input shape follows `(batch_size, sequence_length, input_size)`.

During training, PyTorch handles the recurrent computations and automatic differentiation. We can focus on understanding the architecture, preparing sequences, and evaluating the model.

**Note:** This project is designed to run on a CPU and does not require a GPU.

## Trade-Off: Sequential Memory vs. Long-Term Dependencies

RNNs provide a natural mechanism for processing ordered data, but recurrent computation introduces important trade-offs.

### Advantages

* **Sequential processing:** The model processes inputs in their original order.
* **Hidden-state memory:** Information from previous timesteps can influence later computations.
* **Parameter sharing:** The same recurrent parameters are reused across timesteps.
* **Flexible sequence lengths:** RNNs can process sequences of different lengths, provided the input and training pipeline handle them appropriately.
* **Broad applicability:** RNNs can be applied to time series, text, and other sequential tasks.

### Trade-Offs

* **Sequential computation:** Recurrent dependencies limit parallel computation across timesteps within a sequence.
* **Vanishing gradients:** Gradients can become very small when propagated backward through long sequences.
* **Exploding gradients:** Gradients can also grow excessively, making training unstable.
* **Limited long-term memory:** A basic RNN may struggle to retain information over long intervals.
* **Training challenges:** Performance depends on the sequence structure, model configuration, initialization, and optimization process.

These limitations do not make RNNs useless. They help explain why more specialized architectures, such as Long Short-Term Memory networks (LSTMs), were developed.

**The goal is not simply to process a sequence. It is to understand how recurrent computation carries information forward and why learning long-term dependencies can be difficult.**

## What This Project Explores

* Recurrent neural network architecture
* Sequential data processing
* Hidden-state updates
* Shared recurrent parameters
* Sequence-based prediction
* Backpropagation Through Time
* Loss functions and optimizer-based training

## Technologies

* Python
* PyTorch
* NumPy
* Matplotlib

The exact prediction quality depends on the dataset, sequence-generation process, model architecture, and training configuration.

## Learning Objective

This project continues the series by moving from spatial feature extraction with CNNs to sequential processing with RNNs.

Earlier projects explored artificial neurons, backpropagation, energy-based learning, latent representations, and convolutional feature extraction. Here, we investigate how a neural network can use information from earlier timesteps when processing an ordered sequence.

The aim is to understand both sides:

* **How recurrent networks process sequences using hidden states.**
* **Why learning and preserving long-term dependencies can be challenging.**

## About the Series

**10 Days of Modeling AI for Beginners**

A hands-on series exploring AI concepts through small, understandable projects.

The objective is to learn by building, experimenting, and examining the mechanisms behind AI—not simply by importing a library and accepting its output.

### What's Next?

**Day 9 — Long Short-Term Memory (LSTM)**

We'll explore gated memory mechanisms designed to help recurrent networks learn dependencies across longer sequences.
