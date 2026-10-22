
import os
import json
import gradio as gr

from src.rag_pipeline import RAGPipeline
from src.speech import SarvamSpeech


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(BASE_DIR, "data", "sample_documents.json"),
    "r",
    encoding="utf-8"
) as f:
    documents = json.load(f)


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

if not SARVAM_API_KEY:
    raise ValueError("SARVAM_API_KEY is not set.")


print("Loading RAG pipeline...")

pipeline = RAGPipeline(
    documents,
    GEMINI_API_KEY
)

speech = SarvamSpeech(
    SARVAM_API_KEY
)

print("RAG pipeline ready.")


def ask_text(query):

    if not query.strip():
        return "", "Please enter a question."

    result = pipeline.answer(
        query,
        top_k=3
    )

    context = "\n\n".join(
        f"Result {i+1} | Score: {r['score']:.4f}\n{r['text']}"
        for i, r in enumerate(result["context"])
    )

    return context, result["answer"]


def ask_voice(audio_path):

    if audio_path is None:
        return "", "", "Please record a question.", None

    try:

        transcript = speech.transcribe(
            audio_path
        )

        result = pipeline.answer(
            transcript,
            top_k=3
        )

        context = "\n\n".join(
            f"Result {i+1} | Score: {r['score']:.4f}\n{r['text']}"
            for i, r in enumerate(result["context"])
        )

        answer = result["answer"]

        output_audio = speech.speak(
            answer,
            "/tmp/answer.wav"
        )

        return (
            transcript,
            context,
            answer,
            output_audio
        )

    except Exception as e:

        return (
            "",
            "",
            f"Error: {str(e)}",
            None
        )


with gr.Blocks(
    title="Voice RAG AI"
) as demo:

    gr.Markdown(
        """
        # 🎙️ Voice RAG AI

        ### Voice-Enabled Retrieval-Augmented Generation

        **Sarvam STT → Multi-Strategy Chunking → Embeddings → FAISS → Gemini → Sarvam TTS**
        """
    )

    with gr.Tab("🎤 Voice Question"):

        voice = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record your question"
        )

        voice_button = gr.Button(
            "🚀 Ask with Voice"
        )

        transcript = gr.Textbox(
            label="📝 Sarvam Transcription"
        )

        voice_context = gr.Textbox(
            label="🔎 Retrieved Context",
            lines=8
        )

        voice_answer = gr.Textbox(
            label="🤖 AI Answer",
            lines=5
        )

        voice_audio = gr.Audio(
            label="🔊 AI Voice Answer",
            autoplay=True
        )

        voice_button.click(
            ask_voice,
            inputs=voice,
            outputs=[
                transcript,
                voice_context,
                voice_answer,
                voice_audio
            ]
        )

    with gr.Tab("💬 Text Question"):

        question = gr.Textbox(
            label="Question",
            placeholder="What is machine learning?"
        )

        text_button = gr.Button(
            "🚀 Ask"
        )

        text_context = gr.Textbox(
            label="🔎 Retrieved Context",
            lines=8
        )

        text_answer = gr.Textbox(
            label="🤖 AI Answer",
            lines=5
        )

        text_button.click(
            ask_text,
            inputs=question,
            outputs=[
                text_context,
                text_answer
            ]
        )


if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        ),
        share=True
    )
    
