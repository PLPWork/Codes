import streamlit as st
import speech_recognition as sr
import emoji
from textblob import TextBlob as blob
r=sr.Recognizer()

st.header("Real Time Speech-to-Text")
st.title("Voice to text")
#Realtime
menu=["Microphone","Recorded file"]
def main():
    menu=["Microphone","Recorded file"]
    choice=st.sidebar.selectbox("Menu",menu)
    st.write(emoji.emojize('Everyone :red_heart: Streamlit ',use_aliases=True))
    
    if choice=="Microphone":
        if st.button("Say"):
            with sr.Microphone() as source:
                print('Say something..')
                st.write('Say something')
                audio=r.listen(source,timeout=10)
            try:
                text=r.recognize_google(audio)
                tb=blob(text)
                print("You were saying..")
                print(text)
                #st.write(text)
                print(tb.sentiment)
                #st.write(tb.sentiment)
                result = tb.sentiment.polarity
                
                if result > 0.0:
                    custom_emoji = ':smile:'
                    st.write(emoji.emojize(custom_emoji,use_aliases=True))
                elif result < 0.0:
                    custom_emoji = ':disappointed:'
                    st.write(emoji.emojize(custom_emoji,use_aliases=True))
                else:
                    st.write(emoji.emojize(':expressionless:',use_aliases=True))
                st.info("Polarity Score is:: {}".format(result))
                
            except:
                print("Sorry try again")
                st.write("Soory try again")
        
        if choice=="Recorded file":
            with sr.AudioFile() as source:
                audio_text=r.record(source)
    
                try:
                    text=r.recognize_google(audio_text)
                    print(text)
                    st.write(text)
                except:
                    print("Sorry try again")
                    st.write("Soory try again")
    



    
if __name__=='__main__':
    main()