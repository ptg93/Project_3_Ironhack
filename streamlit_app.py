__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import time
import streamlit as st
from dotenv import load_dotenv

import yt_dlp
from moviepy.editor import VideoFileClip
import whisper
from pyannote.audio import Pipeline
import torch
import ffmpeg

from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

from langchain.chains import RetrievalQA

from langgraph.checkpoint import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
ELEVEN_API_KEY = os.getenv('ELEVEN_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")
elif not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY environment variable not set")

os.makedirs('uploads', exist_ok=True)
os.makedirs('downloads', exist_ok=True)

def save_uploaded_file(uploaded_file):
    start_time = time.time()
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    end_time = time.time()
    duration = end_time - start_time
    st.write(f"File {uploaded_file.name} uploaded successfully to {file_path} in {duration:.2f} seconds")
    return file_path, uploaded_file.name, duration

def extract_metadata_ffmpeg(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        metadata = {
            "duration": float(video_info['duration']),
            "width": int(video_info['width']),
            "height": int(video_info['height']),
            "codec_name": video_info['codec_name'],
            "bit_rate": int(video_info['bit_rate'])
        }
        st.write(f"Extracted metadata: {metadata}")
        return metadata
    except Exception as e:
        st.write(f"Error extracting metadata with ffmpeg: {e}")
        return None

def download_youtube_video(url):
    start_time = time.time()
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': 'downloads/video.%(ext)s',
        'verbose': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            metadata = {
                "title": info_dict.get('title', 'video'),
                "id": info_dict.get('id'),
                "duration": info_dict.get('duration'),
                "upload_date": info_dict.get('upload_date'),
                "uploader": info_dict.get('uploader'),
                "uploader_id": info_dict.get('uploader_id'),
                "view_count": info_dict.get('view_count'),
                "like_count": info_dict.get('like_count'),
                "dislike_count": info_dict.get('dislike_count'),
                "average_rating": info_dict.get('average_rating'),
                "age_limit": info_dict.get('age_limit'),
                "categories": ", ".join(info_dict.get('categories', [])),
                "tags": ", ".join(info_dict.get('tags', [])),
                "ext": info_dict.get('ext'),
                "thumbnail": info_dict.get('thumbnail'),
                "description": info_dict.get('description'),
                "channel": info_dict.get('channel'),
                "channel_id": info_dict.get('channel_id'),
                "is_live": info_dict.get('is_live'),
                "release_date": info_dict.get('release_date'),
                "availability": info_dict.get('availability')
            }

            for key, value in metadata.items():
                if value is None:
                    metadata[key] = "Empty"

            st.write(f"Video title: {metadata['title']}")

            video_ext = metadata['ext']
            initial_path = os.path.abspath(f'downloads/video.{video_ext}')
            if not os.path.isfile(initial_path):
                raise FileNotFoundError(f"Downloaded video file not found: {initial_path}")

            counter = 1
            final_path = os.path.abspath(f'downloads/video_{counter}.{video_ext}')
            while os.path.isfile(final_path):
                counter += 1
                final_path = os.path.abspath(f'downloads/video_{counter}.{video_ext}')

            os.rename(initial_path, final_path)
            st.write(f"Downloaded video saved to: {final_path}")

            end_time = time.time()
            duration = end_time - start_time
            st.write(f"Time taken for downloading video: {duration:.2f} seconds")

            return final_path, metadata, duration
    except Exception as e:
        st.write(f"Error downloading video: {e}")
        return None, None, 0

def extract_audio(video_path):
    start_time = time.time()
    try:
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')
        audio_path = os.path.abspath(audio_path)
        st.write(f"Extracting audio to: {audio_path}")
        video.audio.write_audiofile(audio_path)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Extracted audio file not found: {audio_path}")
        end_time = time.time()
        duration = end_time - start_time
        st.write(f"Audio extracted in {duration:.2f} seconds")
        return audio_path, duration
    except Exception as e:
        st.write(f"Error extracting audio: {e}")
        return None, 0

def select_whisper_model(duration, mode='Fast'):
    if mode == "Accurate":
        if duration > 3600:  # More than 1 hour
            model_name = "tiny"
        elif duration > 1800:  # More than 30 minutes
            model_name = "base"
        elif duration > 600:  # More than 10 minutes
            model_name = "small"
        else:  # 10 minutes or less
            model_name = "medium"
    else:  # Fast mode
        if duration > 1800:  # More than 30 minutes
            model_name = "tiny"
        elif duration > 600:  # More than 10 minutes
            model_name = "base"
        else:  # 10 minutes or less
            model_name = "small"

    st.write(f"Selected Whisper model: {model_name}")
    return model_name

def transcribe_audio(audio_path, duration, mode='Fast'):
    start_time = time.time()
    try:
        st.write(f"Transcribing audio from: {audio_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model_name = select_whisper_model(duration, mode)
        model = whisper.load_model(model_name)

        # Check if CUDA is available and use it
        if torch.cuda.is_available():
            model = model.to("cuda")
            st.write("Using CUDA for Whisper transcription")

        result = model.transcribe(audio_path)
        end_time = time.time()
        transcription_duration = end_time - start_time
        st.write(f"Transcription completed in {transcription_duration:.2f} seconds.")
        return result['text'], transcription_duration
    except Exception as e:
        st.write(f"Error transcribing audio: {e}")
        return "", 0

def transcribe_audio_with_timestamps(audio_path, duration, mode='Fast'):
    start_time = time.time()
    try:
        st.write(f"Transcribing audio from: {audio_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model_name = select_whisper_model(duration, mode)
        model = whisper.load_model(model_name)

        # Check if CUDA is available and use it
        if torch.cuda.is_available():
            model = model.to("cuda")
            st.write("Using CUDA for Whisper transcription")

        result = model.transcribe(audio_path, word_timestamps=True)
        end_time = time.time()
        transcription_duration = end_time - start_time
        st.write(f"Transcription with timestamps completed in {transcription_duration:.2f} seconds.")
        return result, transcription_duration
    except Exception as e:
        st.write(f"Error transcribing audio: {e}")
        return {}, 0

def combine_metadata_and_transcription(metadata, transcription):
    combined_text = "Metadata:\n"
    for key, value in metadata.items():
        combined_text += f"{key}: {value}\n"
    combined_text += "\nTranscription:\n" + transcription
    return combined_text

def combine_metadata_transcription_diarization(metadata, transcription, diarization):
    combined_text = "Metadata:\n"
    for key, value in metadata.items():
        combined_text += f"{key}: {value}\n"
    combined_text += "\nDiarization:\n"
    for segment in diarization.itertracks(yield_label=True):
        speaker = segment[2]
        start_time = segment[0].start
        end_time = segment[0].end
        combined_text += f"Speaker {speaker} [{start_time:.2f} - {end_time:.2f}]: {transcription['segments'][0]['text']}\n"
    return combined_text

def perform_diarization(audio_path, duration, mode='Fast'):
    start_time = time.time()
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_TOKEN"))

        # Check if CUDA is available and use it
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            st.write("Using CUDA for Pyannote diarization")

        diarization_result = pipeline(audio_path)
        end_time = time.time()
        diarization_duration = end_time - start_time
        st.write(f"Diarization completed in {diarization_duration:.2f} seconds.")
        return diarization_result, diarization_duration
    except Exception as e:
        st.write(f"Error during diarization: {e}")
        return None, 0

def process_video(source_type, source, mode='Fast', process_type='Transcription'):
    timings = {}
    total_start_time = time.time()

    if source_type == 'upload':
        video_path, filename, upload_duration = save_uploaded_file(source)
        metadata = extract_metadata_ffmpeg(video_path)
        timings['upload'] = upload_duration
    elif source_type == 'youtube':
        video_path, metadata, download_duration = download_youtube_video(source)
        timings['download'] = download_duration
    else:
        st.write("Invalid source type.")
        return None

    if not video_path or not metadata:
        st.write("Failed to get video or metadata.")
        return None

    audio_path, extract_duration = extract_audio(video_path)
    if not audio_path:
        st.write("Failed to extract audio.")
        return None
    timings['extract_audio'] = extract_duration

    if process_type == 'Transcription':
        transcription, transcribe_duration = transcribe_audio(audio_path, metadata['duration'], mode)
        if not transcription:
            st.write("Failed to transcribe audio.")
            return None
        timings['transcribe_audio'] = transcribe_duration
        combined_text = combine_metadata_and_transcription(metadata, transcription)
    elif process_type == 'Diarization':
        transcription_result, transcribe_duration = transcribe_audio_with_timestamps(audio_path, metadata['duration'], mode)
        if not transcription_result:
            st.write("Failed to transcribe audio with timestamps.")
            return None
        timings['transcribe_audio_with_timestamps'] = transcribe_duration

        diarization_result, diarization_duration = perform_diarization(audio_path, metadata['duration'], mode)
        if not diarization_result:
            st.write("Failed to perform diarization.")
            return None
        timings['diarization'] = diarization_duration

        combined_text = combine_metadata_transcription_diarization(metadata, transcription_result, diarization_result)

    total_end_time = time.time()
    timings['total_processing'] = total_end_time - total_start_time

    st.write(combined_text)
    st.write("Timings:", timings)

    document = Document(page_content=combined_text, metadata=metadata)

    # Clean up files
    try:
        os.remove(video_path)
        os.remove(audio_path)
    except Exception as e:
        st.write(f"Error deleting files: {e}")

    return document

def create_vectorstore(document):
    try:
        vectorstore = Chroma.from_documents(documents=[document], embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        st.write(f"Error creating vectorstore: {e}")
        return None

class VideoChatbot:
    def __init__(self):
        self.retriever = None
        self.qa_chain = None
        self.memory = MemorySaver()
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = '''You are a chatbot that answers questions and performs tasks about a video that the user provides. 
                         Never ask the user to provide a video without first checking if there is one already.
                         If lacking context, assume the user is always talking about the video.
                         First, consider which tools you need to use, if any.
                         When retrieving information, consider that the transcription might not be perfect every time.
                         Then, if relevant, try to identify speakers by their names or usernames, using their dialogue and considering the available metadata.
                         Then use more steps when needed in order to get the right answer. 
                         Finally, you must always identify the language the user is utilizing in their last message and answer in that language, unless the user tells you otherwise.
                      '''
        self.agent = None

    def initialize_qa_chain(self):
        llm = ChatOpenAI(model="gpt-4o")
        try:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever
            )
            st.write("QA chain initialized successfully.")
            self.qa_chain = qa
        except Exception as e:
            st.write(f"Error initializing QA chains: {e}")
            self.qa_chain = None

    def create_agent(self):
        try:
            tools = [
                Tool(
                    name='video_transcript_retriever',
                    func=self.qa_chain.run,
                    description=(
                        'Searches and returns excerpts from the transcript of the user uploaded video.'
                    )
                ),
            ]
            self.agent = create_react_agent(self.model, tools=tools, messages_modifier=self.prompt, checkpointer=self.memory)
            st.write("Agent created successfully.")
        except Exception as e:
            st.write(f"Error creating agent: {e}")

    def process_query(self, query):
        if not self.agent:
            st.write("Agent not initialized.")
            return

        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": "2"}}
        stream = self.agent.stream(inputs, config=config, stream_mode="values")
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                st.write(message)
            else:
                st.write(message.pretty_print())

chatbot = VideoChatbot()

st.title("Video Processing App")

source_type = st.selectbox('Source Type', ['upload', 'youtube'], key='source_type')
if source_type == 'upload':
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"], key='uploaded_file')
else:
    url = st.text_input("YouTube URL", key='youtube_url')

mode = st.selectbox('Mode', ["Fast", "Accurate"], key='mode')
process_type = st.selectbox('Process Type', ["Transcription", "Diarization"], key='process_type')

if st.button("Process Video"):
    if source_type == 'upload' and uploaded_file:
        document = process_video(source_type, uploaded_file, mode, process_type)
    elif source_type == 'youtube' and url:
        document = process_video(source_type, url, mode, process_type)
    else:
        st.write("Please provide a YouTube URL or upload a file.")
        document = None

    if document:
        try:
            vectorstore = Chroma.from_documents(documents=[document], embedding=OpenAIEmbeddings())
            chatbot.retriever = vectorstore.as_retriever()
        except Exception as e:
            st.write(f"Error creating vectorstore: {e}")
            document = None

        if chatbot.retriever:
            st.write("Vectorstore created and retriever initialized.")
            chatbot.initialize_qa_chain()
            chatbot.create_agent()
        else:
            st.write("Failed to create vectorstore.")

query = st.text_input("Enter your query about the video", key='query')
if st.button("Submit Query") and query:
    if chatbot.agent:
        chatbot.process_query(query)
    else:
        st.write("Agent not initialized.")

