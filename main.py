from model import train_and_evaluate

if __name__ == "__main__":
    accuracy, model = train_and_evaluate()
    print(f"Synthetic test accuracy: {accuracy:.2%}")
