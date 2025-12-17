import random
from collections import defaultdict

path = r"C:\Users\ARSHATH ABDULLA A\OneDrive\Desktop\Internships\CODSOFT\CODSOFT\Task5_HANDWRITTEN TEXT\words.txt"

text_data = []

with open(path, encoding="utf-8", errors="ignore") as file:
    for line in file:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        word = parts[-1]
        if word.isalpha():
            text_data.append(word.lower())

corpus = " ".join(text_data)

sequence_length = 4
model = defaultdict(list)

for i in range(len(corpus) - sequence_length):
    key = corpus[i:i + sequence_length]
    next_char = corpus[i + sequence_length]
    model[key].append(next_char)

def generate_text(start_text, length=400):
    result = start_text.lower()
    for _ in range(length):
        key = result[-sequence_length:]
        possible_chars = model.get(key)
        if not possible_chars:
            break
        result += random.choice(possible_chars)
    return result

print("\nGenerated Handwritten-Style Text:\n")
print(generate_text("the government"))
