from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# utilizes gpt2 to train the model

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def train(model, tokenizer, text, epochs=3, learning_rate=1e-5):
    # train the model based on biden's speech.
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

def get_path(env_var, prompt):
    path = os.getenv(env_var)
    if not path:
        path = input(prompt)
    return path

def main():
    model_path = get_path('hanwha_model_path', 'Enter the model save path: ')
    training_data_path = get_path('hanwha_training_data', 'Enter the training data path: ')

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    speech_text = load_text(training_data_path)
    train(model, tokenizer, speech_text)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

if __name__ == "__main__":
    main()
