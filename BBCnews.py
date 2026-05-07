import pandas as pd
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
df = None

# =========================
# SELECT CSV FILE
# =========================
def browse_file():
    global selected_file, df

    selected_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    file_label.config(text=selected_file)

    try:
        df = pd.read_csv(selected_file)

        # Fill column dropdown
        column_menu['menu'].delete(0, 'end')
        for col in df.columns:
            column_menu['menu'].add_command(label=col, command=lambda value=col: selected_column.set(value))

        selected_column.set(df.columns[0])

        preview_text.delete(1.0, END)
        preview_text.insert(END, str(df.head()))

    except Exception as e:
        messagebox.showerror("Error", str(e))

# =========================
# RUN MODEL
# =========================
def run_model():
    global df

    if df is None:
        messagebox.showerror("Error", "Please select a dataset first!")
        return

    try:
        status_label.config(text="Processing...", fg="blue")
        window.update()

        column = selected_column.get()
        documents = df[column].astype(str).tolist()

        stop_words = set(stopwords.words('english'))
        processed_docs = []

        for doc in documents:
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
            passes=10,  # reduced for speed
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
window.title("BBC News Topic Modeling System")
window.geometry("900x700")
window.configure(bg="#eef2f7")

# Title
Label(window, text="BBC Topic Modeling (LDA)", font=("Arial", 18, "bold"), bg="#eef2f7").pack(pady=10)

# File selection
Button(window, text="Select CSV File", command=browse_file, bg="#007BFF", fg="white").pack(pady=5)
file_label = Label(window, text="No file selected", bg="#eef2f7", fg="gray")
file_label.pack()

# Column selection
frame1 = Frame(window, bg="#eef2f7")
frame1.pack(pady=10)

Label(frame1, text="Select Text Column:", bg="#eef2f7").pack(side=LEFT)

selected_column = StringVar()
column_menu = OptionMenu(frame1, selected_column, "")
column_menu.pack(side=LEFT, padx=5)

# Topic number
frame2 = Frame(window, bg="#eef2f7")
frame2.pack(pady=5)

Label(frame2, text="Number of Topics:", bg="#eef2f7").pack(side=LEFT)
topic_entry = Entry(frame2, width=5)
topic_entry.insert(0, "4")
topic_entry.pack(side=LEFT, padx=5)

# Run button
Button(window, text="Run Topic Modeling", command=run_model, bg="green", fg="white", width=25).pack(pady=10)

# Status
status_label = Label(window, text="", bg="#eef2f7")
status_label.pack()

# Preview
Label(window, text="Dataset Preview:", bg="#eef2f7").pack()
preview_text = scrolledtext.ScrolledText(window, width=100, height=10)
preview_text.pack(pady=5)

# Output
Label(window, text="Topics Output:", bg="#eef2f7").pack()
output_text = scrolledtext.ScrolledText(window, width=100, height=15)
output_text.pack(pady=10)

window.mainloop()