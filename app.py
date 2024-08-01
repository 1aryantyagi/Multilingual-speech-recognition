from flask import Flask, render_template, request, redirect, send_file
import main
import os
import io

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    transcription_text = ""
    summary_text = ""
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return redirect(request.url)

        file = request.files['audio_file']
        file_path = os.path.join('./audios', file.filename)
        os.makedirs('./audios', exist_ok=True)
        file.save(file_path)
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Transcribe the audio file
            transcription_text = main.transcribe_audio(file_path)

            # Summarize the transcription
            summary_text = main.summarize_text(transcription_text)
        os.remove(file_path)
    return render_template('index.html', transcription=transcription_text, summary=summary_text)


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    transcription = request.args.get('transcription', '')
    summary = request.args.get('summary', '')

    if not transcription and not summary:
        return redirect('/')

    pdf_path = main.generate_pdf(transcription, summary)
    response = send_file(pdf_path, as_attachment=True)
    response.call_on_close(lambda: os.remove(pdf_path))

    return response


if __name__ == '__main__':
    app.run(debug=True)
