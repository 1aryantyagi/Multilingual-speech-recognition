from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from openai import OpenAI
from fpdf import FPDF

load_dotenv()


def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as file:
            model = OpenAI()
            transcription = model.audio.transcriptions.create(
                model="whisper-1", file=file)
            transcription_text = transcription.text

    except Exception as e:
        transcription_text = f"An error occurred: {e}"

    return transcription_text


def translate_text(text):
    try:
        prompt_template = """Translate the following a text into English. 
        If the text is already in english then return the original text.

        "{text}"
        """

        prompt = PromptTemplate.from_template(prompt_template)

        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-16k",
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        summary = llm_chain.invoke(text)

        return summary.get('text')

    except ImportError as e:
        return f"Error: Missing required package. {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def summarize_text(text1):
    text = translate_text(text1)
    try:
        prompt_template = """Write a concise summary of the following text:

        "{text}"

        CONCISE SUMMARY:
        """

        prompt = PromptTemplate.from_template(prompt_template)

        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-16k",
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        summary = llm_chain.invoke(text)

        return summary.get('text')

    except ImportError as e:
        return f"Error: Missing required package. {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def generate_pdf(transcription, summary):
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Audio Processing Transcript", ln=True, align='C')

        pdf.multi_cell(
            0, 10, txt=f"Transcription:\n{transcription}\n\nSummary:\n{summary}")

        pdf_output_path = "transcript.pdf"
        pdf.output(pdf_output_path)

        return pdf_output_path

    except ImportError as e:
        return f"Error: Missing required package. {e}"
    except Exception as e:
        return f"An error occurred: {e}"
