import argparse
import torch
import pandas as pd
import os
from datasets import Audio, Dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sentence_segment import SyllableSegmentation
from utils import convert_mp4_to_wav, perform_vad, generate_srt, burn_srt_to_video
from pydub import AudioSegment


def convert_audio_to_wav(audio_file, target_sr):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    output_wav_file = audio_file.rsplit('.', 1)[0] + "_converted.wav"
    audio.export(output_wav_file, format="wav")
    return output_wav_file

def main(args):
    SAMPLING_RATE = 16000

    # Do ASR
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_path,
        chunk_length_s=30,
        device=device,
        torch_dtype=torch.float16,
    )

    audio_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.mp4', '.wav', '.mp3'))]

    for audio_file in audio_files:
        if audio_file.endswith('.mp4'):
            wav_file = convert_mp4_to_wav(audio_file)
        elif audio_file.endswith('.wav'):
            # Check sampling rate and convert if necessary
            audio = AudioSegment.from_wav(audio_file)
            if audio.frame_rate != SAMPLING_RATE:
                wav_file = convert_audio_to_wav(audio_file, SAMPLING_RATE)
            else:
                wav_file = audio_file
        else:  # Assuming other audio formats such as .mp3, etc.
            wav_file = convert_audio_to_wav(audio_file, SAMPLING_RATE)

        _, chunklist = perform_vad(wav_file, 'temp_directory_for_chunks')

        # for faster inference, create dataset
        audio_dataset = Dataset.from_dict({"audio": [c["fname"] for c in chunklist]}).cast_column("audio", Audio())

        prediction_gen = pipe(
            KeyDataset(audio_dataset, "audio"),
            generate_kwargs={"task": "transcribe", "language": "Thai"},
            return_timestamps=False,
            batch_size=4,
            ignore_warning=True,
        )

        predictions = [out for out in prediction_gen]

        vad_transcriptions = {"start": [], "end": [], "prediction": []}

        for vad_chunk, pred in zip(chunklist, predictions):
            start_in_samples, end_in_samples = vad_chunk["start"], vad_chunk["end"]
            start_in_s = start_in_samples / (SAMPLING_RATE)
            end_in_s = end_in_samples / (SAMPLING_RATE)

            vad_transcriptions["prediction"].append(pred["text"])
            vad_transcriptions["start"].append(start_in_s)
            vad_transcriptions["end"].append(end_in_s)

        ss = SyllableSegmentation()
        uncorrected_segments = ss(vad_transcriptions=vad_transcriptions, segment_duration=4.0)

        base_filename = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = f"{args.output_folder}/{base_filename}.{args.output_format}"

        if args.output_format == 'csv':
            df = pd.DataFrame(uncorrected_segments)
            df.to_csv(output_file, index=False)
        elif args.output_format == 'srt':
            generate_srt(uncorrected_segments, output_file)
            if args.burn_srt:
                burn_srt_to_video(audio_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR pipeline with options")
    parser.add_argument("--input_folder", required=True, help="Input folder path containing audio files")
    parser.add_argument("--output_folder", required=True, help="Output folder path to save the results")
    parser.add_argument("--model_path", default='/path/to/default/model', help="Path to the whisper model")
    parser.add_argument("--output_format", choices=['csv', 'srt'], default='csv', help="Output format, either csv or srt")
    parser.add_argument("--burn_srt", action='store_true', help="Option to burn the srt to the input video (only works if output_format is srt)")

    args = parser.parse_args()
    main(args)
