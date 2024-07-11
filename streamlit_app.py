import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

def save_uploaded_file(uploaded_file):
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join("tempDir", uploaded_file.name)

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def store_transcript(transcript):
    vectorstore = PineconeVectorStore.from_texts(
        [transcript],
        embeddings,
        index_name=index_name
    )
    return vectorstore

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "video-transcriptions"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings()

# Initialize LangChain components
llm = OpenAI()
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="You are a helpful assistant. Answer the following question based on the context: {context}\n\nQuestion: {question}\n\nAnswer:"
)
qa_chain = RetrievalQA(
    retriever=None,  # This will be set after storing the transcript
    llm=llm,
    prompt_template=prompt_template
)

memory = ConversationBufferMemory()
conversation_chain = ConversationChain(llm=llm, memory=memory)

# Streamlit App
st.title("Video Transcription Chatbot")

option = st.selectbox('How would you like to upload the video?', ('Upload from device', 'YouTube link'))

file_path = None

if option == 'Upload from device':
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
else:
    youtube_link = st.text_input("Enter YouTube link")
    # You would need to add code here to download the video from YouTube if using this option

if file_path:
    audio_path = extract_audio(file_path)
    transcript = transcribe_audio(audio_path)
    st.write(transcript)
    vectorstore = store_transcript(transcript)
    qa_chain.retriever = vectorstore.as_retriever()

    user_query = st.text_input("Ask a question about the video:")
    if user_query:
        answer = qa_chain.run({"question": user_query, "context": transcript})
        st.write(answer)

    user_query = st.text_input("Continue the conversation:")
    if user_query:
        answer = conversation_chain.run(input=user_query)
        st.write(answer)
