from model import train_and_evaluate

acc, model = train_and_evaluate()
print(f"Precision of the biometric classifier: {acc*100:.2f}%")

