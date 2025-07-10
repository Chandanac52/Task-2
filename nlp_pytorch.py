import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ----------------------
# 1. Load Data from reviews.txt
# ----------------------
texts = []
labels = []

with open("reviews.txt", "r", encoding="utf-8") as file:
    for line in file:
        if '\t' in line:
            label, sentence = line.strip().split('\t', 1)
            labels.append(int(label))
            texts.append(sentence)

# ----------------------
# 2. Preprocessing with CountVectorizer
# ----------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = torch.tensor(labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# ----------------------
# 3. Define Deep Learning Model
# ----------------------
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)  # 2 output classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

model = SentimentClassifier(input_dim=X.shape[1])

# ----------------------
# 4. Training the Model
# ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = []
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ----------------------
# 5. Evaluate Accuracy
# ----------------------
model.eval()
with torch.no_grad():
    preds = model(X_test)
    predicted = torch.argmax(preds, dim=1)
    acc = accuracy_score(y_test, predicted)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# ----------------------
# 6. Plot Training Loss
# ----------------------
plt.plot(train_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss_graph.png")
plt.show()

# ----------------------
# 7. Save the Model
# ----------------------
torch.save(model.state_dict(), "sentiment_model.pth")
print("\nModel saved as sentiment_model.pth and training_loss_graph.png")
