import os
import nltk
from tkinter import *
from tkinter import filedialog, scrolledtext, messagebox
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
import gensim

# =========================
# NLTK SETUP
# =========================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

selected_file = ""

# =========================
# FILE BROWSER
# =========================
def browse_file():
    global selected_file
    selected_file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    file_label.config(text=selected_file)

# =========================
# RUN MODEL
# =========================
def run_model():
    if not selected_file:
        messagebox.showerror("Error", "Please select a file first!")
        return

    try:
        status_label.config(text="Processing...", fg="blue")
        window.update()

        with open(selected_file, "r", encoding="utf-8") as file:
            text = file.read()

        documents = text.split("\n")

        stop_words = set(stopwords.words('english'))
        processed_docs = []

        for doc in documents:
            doc = doc.strip()
            if doc:
                tokens = word_tokenize(doc.lower())
                filtered = [
                    word for word in tokens
                    if word.isalnum() and word not in stop_words
                ]
                if filtered:
                    processed_docs.append(filtered)

        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        num_topics = int(topic_entry.get())

        lda_model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=15,
            random_state=42
        )

        output_text.delete(1.0, END)
        output_text.insert(END, "===== TOPICS =====\n\n")

        for idx, topic in lda_model.print_topics():
            output_text.insert(END, f"Topic {idx}:\n{topic}\n\n")

        status_label.config(text="Completed ✅", fg="green")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="Error ❌", fg="red")

# =========================
# GUI DESIGN
# =========================
window = Tk()
window.title("Advanced Topic Modeling System")
window.geometry("800x600")
window.configure(bg="#f4f4f4")

title = Label(window, text="Topic Modeling (LDA)", font=("Arial", 18, "bold"), bg="#f4f4f4")
title.pack(pady=10)

# File selection
browse_btn = Button(window, text="Select Text File", command=browse_file, bg="#007BFF", fg="white")
browse_btn.pack(pady=5)

file_label = Label(window, text="No file selected", bg="#f4f4f4", fg="gray")
file_label.pack()

# Topic number input
topic_frame = Frame(window, bg="#f4f4f4")
topic_frame.pack(pady=10)

Label(topic_frame, text="Number of Topics:", bg="#f4f4f4").pack(side=LEFT)
topic_entry = Entry(topic_frame, width=5)
topic_entry.insert(0, "4")
topic_entry.pack(side=LEFT, padx=5)

# Run button
run_btn = Button(window, text="Run Topic Modeling", command=run_model, bg="green", fg="white", width=20)
run_btn.pack(pady=10)

# Status
status_label = Label(window, text="", bg="#f4f4f4", font=("Arial", 10))
status_label.pack()

# Output area
output_text = scrolledtext.ScrolledText(window, width=90, height=20)
output_text.pack(pady=10)

window.mainloop()