import torch

from core.rnn import RNN
from training.dataset import create_dataloaders
from training.train import Trainer


def main():
    # Reproducibility
    torch.manual_seed(42)

    # Configuration
    BATCH_SIZE = 32
    MAX_VOCAB_SIZE = 10000
    MAX_LENGTH = 80
    EMBEDDING_DIM = 64
    HIDDEN_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Load IMDb dataset
    train_loader, test_loader, vocabulary = create_dataloaders(
        batch_size=BATCH_SIZE,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_length=MAX_LENGTH,
    )

    # Initialize model
    model = RNN(
        vocab_size=len(vocabulary),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
    )

    print("\nModel initialized.")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        learning_rate=LEARNING_RATE,
    )

    # Train and evaluate
    history = trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
    )

    # Save model and vocabulary
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "config": {
                "embedding_dim": EMBEDDING_DIM,
                "hidden_size": HIDDEN_SIZE,
                "max_length": MAX_LENGTH,
            },
        },
        "rnn_imdb.pth",
    )

    print("\nTraining complete!")
    print("Model saved to rnn_imdb.pth")

    print(f"\nFinal test accuracy: {history['test_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()

# Note: Maximum review length reduced from 200 tokens into 80 tokens to avoid padding in first place.
# But this may lead to loss of information in some reviews. Consider increasing the max_length for farther experiments.
