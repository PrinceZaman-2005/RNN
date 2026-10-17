# Training Notes

## 1. Training Pipeline

The model learns to classify IMDb movie reviews as positive or negative.

```text
Token IDs → Embedding → RNN → Linear Layer → Logits
                                              ↓
                                         Loss Function
                                              ↓
                                         Backpropagation
                                              ↓
                                         Weight Updates
```

## 2. Loss Function

We use `BCEWithLogitsLoss()` for binary classification.

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, labels)
```

It combines sigmoid and binary cross-entropy in a numerically stable operation. Pass raw logits directly to the loss function.

## 3. Backpropagation

```python
optimizer.zero_grad()
logits = model(inputs)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

* `zero_grad()` clears previous gradients.
* `loss.backward()` calculates gradients through the model, including the embedding and RNN layers.
* `optimizer.step()` updates trainable parameters using Adam.

## 4. Predictions

Apply sigmoid to convert logits into probabilities.

```python
probabilities = torch.sigmoid(logits)
predictions = (probabilities >= 0.5).long()
```

* **Probability ≥ 0.5:** Positive (`1`)
* **Probability < 0.5:** Negative (`0`)

## 5. Evaluation

The trainer tracks:

* **Training loss:** How well the model fits training data.
* **Test loss:** Prediction error on unseen data.
* **Training accuracy:** Percentage of training reviews classified correctly.
* **Test accuracy:** Percentage of test reviews classified correctly.

Use `model.eval()` and `torch.no_grad()` during evaluation.

## 6. Limitations

* Standard RNNs can struggle with long-term dependencies.
* Padding tokens can affect the final hidden state.
* Unknown words map to `<UNK>`.
* Accuracy and training time must be measured experimentally.
