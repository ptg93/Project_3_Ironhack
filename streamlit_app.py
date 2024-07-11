import streamlit as st
import whisper
import subprocess
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

def save_uploaded_file(uploaded_file):
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")
    file_path = os.path.join("tempDir", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_audio(video_path):
    audio_path = "audio.wav"
    command = [
        "ffmpeg",
        "-y",  # Add this flag to automatically overwrite files
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred while extracting audio: {e.stderr}")
    return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def store_transcript(transcript):
    try:
        vectorstore = PineconeVectorStore.from_texts(
            [transcript],
            embeddings,
            index_name=index_name
        )
        return vectorstore
    except Exception as e:
        st.error(f"An error occurred while storing the transcript: {str(e)}")
        raise e

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

memory = ConversationBufferMemory()

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
    # Add code here to download the video from YouTube if using this option

if file_path:
    audio_path = extract_audio(file_path)
    transcript = transcribe_audio(audio_path)
    st.write(transcript)
    vectorstore = store_transcript(transcript)

    qa_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        memory=memory,
        input_variables=["context", "question"],
        template=prompt_template
    )

    user_query = st.text_input("Ask a question about the video:")
    if user_query:
        answer = qa_chain.run({"question": user_query, "context": transcript})
        st.write(answer)

    user_query = st.text_input("Continue the conversation:")
    if user_query:
        answer = qa_chain.run({"question": user_query})
        st.write(answer)
