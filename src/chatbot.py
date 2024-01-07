import openai
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_speech(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_relevant_section(speech, question):
    # extracts relevant keywords/section from the question.
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(question.lower())
    keywords = [word for word in word_tokens if word not in stop_words]
    sentences = sent_tokenize(speech)
    relevant_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return ' '.join(relevant_sentences)[:1000] if relevant_sentences else ""

def remove_repetitions(text):
    # removes repeated sentences to generate more concise answer.
    sentences = sent_tokenize(text)
    seen = set()
    non_repetitive_sentences = []

    for sentence in sentences:
        if sentence not in seen:
            non_repetitive_sentences.append(sentence)
            seen.add(sentence)

    return ' '.join(non_repetitive_sentences)

def ask_question(question, context):
    if not context:
        # If no context is found, directly ask GPT-3 based on general knowledge
        prompt = f"Provide a direct, concise answer to the following question based on general knowledge. Avoid asking additional questions. :\n\n{question}"
    else:
        # context was found. generate answer based on biden's speech.
        prompt = f"Based on the following excerpt from Biden's speech, provide a concise answer to the question. Avoid asking additional questions or making unrelated comments. :\n\n{context}\n\nQuestion: {question}\nAnswer:"  
    try:
        # generate answer
        response = openai.Completion.create(
            engine="davinci", 
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            stop=['\n']
        )
        processed_answer = remove_repetitions(response.choices[0].text.strip())
        return processed_answer
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def main():
    speech_path = 'hanwha_training_data'
    speech = load_speech(speech_path)

    while True:
        question = input("Ask a question (type 'quit' to quit): ")
        if question.lower() == 'quit':
            break
        context = extract_relevant_section(speech, question)
        answer = ask_question(question, context)
        print("Answer:", answer)

if __name__ == "__main__":
    main()

