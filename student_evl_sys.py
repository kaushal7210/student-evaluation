import tkinter as tk
import tkinter.messagebox as messagebox
import speech_recognition as sr
import threading
import pyaudio
import wave
import os
import pickle

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
model=SentenceTransformer('paraphrase-MiniLM-L6-v2')

class SpeechToTextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speech to text Conveter")
        
        ds=pickle.load(open('preprocess_data.pkl','rb'))
        
        q = ds['question']
        self.question_text = tk.Text(self, height=10, width=100)
        self.question_text.insert(tk.END,q)        
        self.question_text.pack(pady=20)
        
        self.record_button=tk.Button(self,
                                     text="Start Recording",
                                     command=self.start_recording)
        self.record_button.pack(pady=20)

        
        self.stop_button=tk.Button(self,
                                     text="Stop Recording",
                                     command=self.stop_recording,state=tk.DISABLED)
        self.stop_button.pack(pady=20)

        self.play_button=tk.Button(self,
                                     text="Play Recording",
                                     command=self.play_recording,state=tk.DISABLED)
        self.play_button.pack(pady=20)

        self.convert_button=tk.Button(self,
                                     text="Convert to Text",
                                     command=self.convert_audio_to_text,state=tk.DISABLED)
        self.convert_button.pack(pady=20)
        self.output_text = tk.Text(self, height=15, width=100)
        self.output_text.pack(pady=20)

        self.audio_file_path="recorded_audio.wav"
        self.recording=False

    def preprocess_text(self,text):
        text=text.lower()
        words=word_tokenize(text)
        stop_word=set(stopwords.words('english'))
        words=[word for word in words if word not in stop_word]
        stemmer=PorterStemmer()
        words=[stemmer.stem(word) for word in words]

        proprocessed_text=' '.join(words)
        return proprocessed_text

    def suggest_sections(self,ans,ds,min_suggestions=1):
        ds=pickle.load(open('preprocess_data.pkl','rb'))
        preprocessed_ans = self.preprocess_text(ans)
        ans_embedding = model.encode(preprocessed_ans)
        section_embedding = model.encode(ds['ans1'].tolist())
        similarities = util.pytorch_cos_sim(ans_embedding,section_embedding)[0]
        similarity_threhold = 0.2
        relevant_indices = []
        while len(relevant_indices)<min_suggestions and similarity_threhold>0:
            relevant_indices = [i for i, sim in enumerate(similarities)if sim>similarity_threhold]
            similarity_threhold -= 0.5 #st=st-0.5
            sorted_indices = sorted(relevant_indices,key=lambda i: similarities[i],reverse=True)
            suggestions = [
            {
                'index': i,
                'question': ds.iloc[i]['question'],
                'ans': ds.iloc[i]['ans'],
                'similarity_score': similarities[i].item()
            
            }
            for i in sorted_indices
        ]
        return suggestions


    def start_recording(self):
        self.recording=True
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.play_button.config(state=tk.DISABLED)
        self.convert_button.config(state=tk.DISABLED)

        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,channels=1,
                                    rate=44100,input=True,frames_per_buffer=1024)
        self.frames=[]
        self.recording_thread=threading.Thread(target=self.record)
        self.recording_thread.start()

    def record(self):
        while self.recording:
            data=self.stream.read(1024)
            self.frames.append(data)
    def stop_recording(self):
        self.recording=False
        self.stream.stop_stream
        self.stream.close()
        self.audio.terminate()

        wf=wave.open(self.audio_file_path,'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(self.frames))
        wf.close()

        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.NORMAL)
        self.convert_button.config(state=tk.NORMAL)

    def play_recording(self):
        os.system(f"Start {self.audio_file_path}")
    def convert_audio_to_text(self):
        r=sr.Recognizer()
        with sr.AudioFile(self.audio_file_path) as source:
            audio_data=r.record(source)
            try:
                text=r.recognize_google(audio_data)
                #messagebox.showinfo("Speech to text",text)
                ds=pickle.load(open('preprocess_data.pkl','rb'))

            
                suggestions = self.suggest_sections(text, ds, min_suggestions = 1)
                n=1
                suggestion = suggestions[n-1]
                if suggestion:
                    output= (f"sr_no: {suggestion['index']+1}\n"
                    f"question: {suggestion['question']}\n"
                    f"Your ans : {text}\n"\
                    f"Expected ans: {suggestion['ans']}\n"
                    f"similarity score: {round(suggestion['similarity_score']*100,2)}%\n")
                    self.output_text.insert(tk.END,output)
                    
                    # print("_________________________________________________________________________________________\n")
                else:
                    print("No record is found")

            except sr.UnknownValueError:
                messagebox.showwarning("Speech to text"," Could not understand the audio")
            except sr.RequestError as e:
                messagebox.showerror("Speech to text",f"Error occurred : {e}")

if __name__=="__main__":
    app=SpeechToTextApp()
    app.mainloop()

    
