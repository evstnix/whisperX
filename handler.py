import os, io, json, base64, tempfile, time, math
align_model_name = p.get("align_model") # явный выбор wav2vec2 модели
return_raw = bool(p.get("return_raw", True))


return_srt = bool(p.get("return_srt", True))
return_vtt = bool(p.get("return_vtt", False))


# whisper overrides (будут проброшены как есть в faster-whisper)
whisper_over = p.get("whisper", {}) or {}
whisper_kwargs = {"batch_size": batch_size}
for k, v in list(whisper_over.items()):
if k in ALLOWED_WHISPER_ARGS:
whisper_kwargs[k] = v


# По умолчанию уменьшим склейки контекста
whisper_kwargs.setdefault("condition_on_previous_text", False)
# И включим VAD-фильтр для более устойчивой сегментации
whisper_kwargs.setdefault("vad_filter", True)


# 1) load audio
audio_path = _download_to_tmp(p)
audio = whisperx.load_audio(audio_path)


# 2) ASR
asr = _ensure_model(model_name, compute_type, batch_size)
result_asr = asr.transcribe(
audio,
language=language,
**whisper_kwargs,
)


segments_raw = result_asr.get("segments", [])
detected_lang = result_asr.get("language") or language


# 3) Alignment (wav2vec2)
segments_aligned = segments_raw
diarize_segments = None


if align:
lang = (language or detected_lang or "ru")
align_model, meta = _ensure_aligner(lang, model_name=align_model_name)
aligned = whisperx.align(
segments_raw, align_model, meta, audio, DEVICE,
return_char_alignments=char_align
)
# align(...) может вернуть dict или уже список
segments_aligned = aligned.get("segments", aligned)


# 4) (Optional) diarization
if diarize:
if not hf_token:
raise ValueError("Diarization requested but no HF token provided (env HF_TOKEN or input.hf_token)")
diar = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
diarize_segments = diar(audio,
min_speakers=p.get("min_speakers"),
max_speakers=p.get("max_speakers"))
segments_aligned = whisperx.assign_word_speakers(diarize_segments, {"segments": segments_aligned})["segments"]


# 5) build outputs
out = {
"device": DEVICE,
"model": model_name,
"compute_type": compute_type,
"language": detected_lang,
"timing": {"total_sec": round(time.time() - t0, 3)}
}


if return_raw:
out["segments_raw"] = segments_raw
out["segments"] = segments_aligned # aligned (with words)


if diarize_segments is not None:
out["diarization"] = diarize_segments


if return_srt:
out["srt"] = _make_srt(segments_aligned)
if return_vtt:
out["vtt"] = _make_vtt(segments_aligned)


try:
os.remove(audio_path)
except Exception:
pass


return out




# Entrypoint
runpod.serverless.start({"handler": handler})