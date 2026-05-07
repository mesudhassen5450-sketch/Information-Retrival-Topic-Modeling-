import nltk
from tkinter import *
from tkinter import ttk, scrolledtext
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.datasets import fetch_20newsgroups
from gensim import corpora
import gensim

# =========================
# NLTK SETUP
# =========================
nltk.download('punkt')
nltk.download('stopwords')

# =========================
# MAIN FUNCTION
# =========================
def run_model():
    status_var.set("Loading dataset...")

    newsgroups = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes')
    )

    documents = newsgroups.data[:500]

    status_var.set("Processing data...")

    stop_words = set(stopwords.words('english'))

    custom_words = {
        "would", "could", "also", "one", "two", "like",
        "know", "get", "use", "even", "good", "well",
        "many", "much", "make", "take", "see", "want",
        "think", "people", "something", "really",
        "anyone", "please", "thanks", "email"
    }

    stop_words = stop_words.union(custom_words)

    processed_docs = []

    for doc in documents:
        tokens = word_tokenize(doc.lower())

        filtered = [
            w for w in tokens
            if w.isalpha()
            and len(w) > 4
            and w not in stop_words
        ]

        if filtered:
            processed_docs.append(filtered)

    status_var.set("Building model...")

    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=10, no_above=0.4)

    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    num_topics = int(topic_entry.get())

    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=25,
        random_state=42
    )

    # OUTPUT
    output_box.delete(1.0, END)
    output_box.insert(END, "=== TOPIC MODEL RESULTS ===\n\n")

    for idx, topic in lda_model.print_topics(num_words=8):
        output_box.insert(END, f"Topic {idx}:\n{topic}\n\n")

    status_var.set("Completed ✔")

# =========================
# WINDOW SETUP (REAL SOFTWARE LOOK)
# =========================
window = Tk()
window.title("AI Topic Modeling Studio")
window.geometry("1000x650")
window.configure(bg="#1e1e2f")

# =========================
# SIDEBAR
# =========================
sidebar = Frame(window, bg="#111827", width=250)
sidebar.pack(side=LEFT, fill=Y)

Label(sidebar, text="📊 Topic Model AI",
      bg="#111827", fg="white",
      font=("Arial", 16, "bold")).pack(pady=20)

Label(sidebar, text="20 Newsgroups LDA System",
      bg="#111827", fg="gray").pack()

Label(sidebar, text="Topics:",
      bg="#111827", fg="white").pack(pady=10)

topic_entry = Entry(sidebar, width=5, font=("Arial", 12))
topic_entry.insert(0, "4")
topic_entry.pack()

Button(sidebar, text="▶ Run Model",
       command=run_model,
       bg="#22c55e", fg="white",
       font=("Arial", 12, "bold"),
       width=18).pack(pady=20)

# STATUS
status_var = StringVar()
status_var.set("Ready")

Label(sidebar, textvariable=status_var,
      bg="#111827", fg="#38bdf8",
      font=("Arial", 10)).pack(pady=10)

# =========================
# MAIN PANEL
# =========================
main_frame = Frame(window, bg="#1e1e2f")
main_frame.pack(side=LEFT, fill=BOTH, expand=True)

Label(main_frame, text="Topic Output Console",
      bg="#1e1e2f", fg="white",
      font=("Arial", 14, "bold")).pack(pady=10)

output_box = scrolledtext.ScrolledText(
    main_frame,
    width=90,
    height=30,
    bg="#0f172a",
    fg="#e2e8f0",
    insertbackground="white",
    font=("Consolas", 10)
)
output_box.pack(padx=10, pady=10)

window.mainloop()