#!/usr/bin/env python3

import parselmouth as pm
import textgrids as tg
from pathlib import Path
import pprint

# Data types for typing annotations
from textgrids import Tier

# TODO: MM:SS.sss formatting
# TODO: 1 vs. 3 point per phoneme contours, and keep it as an option
# TODO: Go through all the labels and build a phoneme bank in one go, then just query it.
# 		Make it a list, so we keep duplicates, and just choose one at random during synth.

# NOTE: Paths are relative to this file.
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / "data")
OUTPUT_DIR = (BASE_DIR / "output")
# NOTE: Unfortunately, parselmouth & co do not handle Path-like objects, so we resolve everything as strings...
SOUND_FILE = (DATA_DIR / "faure.wav").as_posix()
GRID_FILE =  (DATA_DIR / "faure.TextGrid").as_posix()
# Output files
OUTPUT_WAV = (OUTPUT_DIR / "raw_concat.wav").as_posix()
OUTPUT_SYNTHESIZED_WAV =  (OUTPUT_DIR / "espeak-synth.wav").as_posix()
OUTPUT_SYNTHESIZED_GRID = (OUTPUT_DIR / "espeak-synth.TextGrid").as_posix()
OUTPUT_FINAL_WAV =  (OUTPUT_DIR / "result.wav").as_posix()

# Utility functions
def format_duration(seconds: float) -> str:
	"""Returns a MM:SS.sss string for the given input"""
	minutes = seconds // 60
	seconds, ms = divmod(seconds % 60, 1)
	# Round to 3 decimals, without the decimal point
	ms = str(round(ms, 3)).split(".")[1]
	return "{:02d}:{:02d}.{:s}".format(int(minutes), int(seconds), ms)

sound = pm.Sound(SOUND_FILE)
grid = tg.TextGrid(GRID_FILE)
pp = pm.praat.call(sound, "To PointProcess (zeroes)", 1, "yes", "no")
# NOTE: Layer name
diphones = grid["Diphones"]

# Extract a small slice of silence as our initial sound object.
# NOTE: Look into gaussian windows to avoid edge effect...
concatenated_sound = sound.extract_part(0, 0.01, pm.WindowShape.RECTANGULAR, 1, False)
diphones_sound = {}

def extract_diphone(phoneme_1: str, phoneme_2: str, diphones: Tier):
	for j, d in enumerate(diphones[:-1]):
		left = d
		right = diphones[j+1]

		phoneme = left.text
		next = right.text

		#print(f"{phoneme} vs. {phoneme_1} // {next} vs. {phoneme_2}")
		if phoneme == phoneme_1 and next == phoneme_2:
			mid_left = left.mid
			mid_right = right.mid

			# NOTE: we can't choose only risong or descending crossings...
			"""
			mid_left = sound.get_nearest_zero_crossing(mid_left, 1)
			mid_right = sound.get_nearest_zero_crossing(mid_right, 1)
			"""

			# ...So we go through a PointProcess to only keep *rising* zero-crossings
			id = pm.praat.call(pp, "Get nearest index", mid_left)
			mid_left = pm.praat.call(pp, "Get time from index", id)
			id = pm.praat.call(pp, "Get nearest index", mid_right)
			mid_right = pm.praat.call(pp, "Get time from index", id)

			# Return original phoneme data to allow proper PSOLA processing later on...
			diphone_data = (
				{
					"phoneme": phoneme_1,
					"orig_start": left.xmin,
					"orig_end": left.xmax,
					"duration": left.dur,
					"diphone_pos": "left",
					"mid": mid_left,
					"extracted_duration": left.xmax - mid_left,	# i.e., duration / 2
				},
				{
					"phoneme": phoneme_2,
					"orig_start": right.xmin,
					"orig_end": right.xmax,
					"duration": right.dur,
					"diphone_pos": "right",
					"mid": mid_right,
					"extracted_duration": right.xmax - mid_right,
				}
			)
			return (sound.extract_part(mid_left, mid_right, pm.WindowShape.RECTANGULAR, 1, False), diphone_data)
	return (None, None)

# FIXME: Feed the full sentence to espeak, duh'
def synthesize_word(word: str, output_sound):
	print(f"synthesize_word on {word}")
	# TODO: Test voices
	praat_synth = pm.praat.call("Create SpeechSynthesizer", "French (France)", "Female1")
	# Setup espeak to use XSampa
	pm.praat.call(praat_synth, "Speech output settings", 16000, 0.01, 1, 1, 175, "Kirshenbaum_espeak")
	text_synth, sound_synth = pm.praat.call(praat_synth, "To Sound", word, "yes")
	n = pm.praat.call(text_synth, "Get number of intervals", 4)
	# FIXME: Whut? 2 extra intervals with trailing '_:', '_' labels...
	print(f"n: {n}")
	# Quick, nobody noticed...
	# FIXME: Possibly Py3.12 related?
	# FIXME: Nope, borked on 3.11, too... macOS, then?
	if n > 2:
		n -= 2
	# Serialize it to double-check that...
	text_synth.save(OUTPUT_SYNTHESIZED_GRID)
	sound_synth.save(OUTPUT_SYNTHESIZED_WAV, "WAV")

	pitch_synth = pm.praat.call(sound_synth, "To Pitch (shs)", 0.01, 50, 15, 1250, 15, 0.84, 600, 48)

	# Includes empty intervals (word boundaries)
	# FIXME: Speaking of word boundaries... We've only recorded *sentence* boundaries diphones, not *word* boundaries ones...
	#        So we'll have some more diphones to record?
	#        (Because just annotating extra boundaries did not go well w/ faure, c.f., /(a)vɔdka/)
	# See NOTE above, we might just need to feed full sentences to espeak instead of word per word...
	# FIXME: We still get those, so, basically, just drop _ when they're not at the edge of the list.
	espeak_phonemes = [pm.praat.call(text_synth, "Get label of interval", 4, i + 1) for i in range(n)]
	print(f"espeak_phonemes: {espeak_phonemes}")
	# Replace empty phonemes w/ an underscore
	espeak_phonemes = ["_" if x == "" else x for x in espeak_phonemes]
	espeak_transcription = "".join(espeak_phonemes)
	# Sanity check
	print(f"espeak transcription: {espeak_transcription}")

	# Compute the f0 mean of each phoneme (via Praat's "Get Mean")
	espeak_phonemes_start_ts = [pm.praat.call(text_synth, "Get start time of interval", 4, i + 1) for i in range(n)]
	espeak_phonemes_end_ts   = [pm.praat.call(text_synth, "Get end time of interval", 4, i + 1) for i in range(n)]

	# Zip it!
	espeak_data = []
	for phoneme, start, end in zip(espeak_phonemes, espeak_phonemes_start_ts, espeak_phonemes_end_ts):
		mean_f0 = pm.praat.call(pitch_synth, "Get mean", start, end, "Hertz")
		# We'll store everything in a list of dicts...
		d = {
			"phoneme": phoneme,
			"start": float(start),
			"end": float(end),
			"duration": float(end) - float(start),
			"f0": float(mean_f0)
		}
		espeak_data.append(d)

	# FIXME: Beware, espeak might chuck out the IPA `g` in XSampa (i.e., U+0261 (ɡ) instead of U+0067 (g))...
	for i, phoneme in enumerate(espeak_transcription[:-1]):
		phone1 = phoneme
		phone2 = espeak_transcription[i+1]

		extraction, diphone_data = extract_diphone(phone1, phone2, diphones)

		# Concat
		if extraction != None and diphone_data != None:
			# Compute phoneme position in the concatenated stream
			left_pos = output_sound.duration
			espeak_data[i]["concat_start"] = left_pos
			espeak_data[i]["concat_duration"] = diphone_data[0]["extracted_duration"]
			espeak_data[i]["concat_end"] = left_pos + espeak_data[i]["concat_duration"]

			right_pos = espeak_data[i]["concat_end"]
			espeak_data[i+1]["concat_start"] = right_pos
			espeak_data[i+1]["concat_duration"] = diphone_data[1]["extracted_duration"]
			espeak_data[i+1]["concat_end"] = right_pos + espeak_data[i+1]["concat_duration"]
			# Sanity check...
			# NOTE:
			#	- In diphone_data:
			#		- orig ts are ts in the original recording
			#	- In espeak data:
			#		- unprefixed ts are ts in the espeak synth
			#		- concat ts are the output ts in the concatenated stream
			pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)
			print("diphone_data (left):")
			pp.pprint(diphone_data[0])
			print("espeak_data (left):")
			pp.pprint(espeak_data[i])
			print("diphone_data (right):")
			pp.pprint(diphone_data[1])
			print("espeak_data (right):")
			pp.pprint(espeak_data[i+1])

			output_sound = output_sound.concatenate([output_sound, extraction])
		else:
			print(f"Failed to extract phoneme {phoneme}")
	return (output_sound, espeak_data)

# FIXME: Or sys.argv[1]
sentence = "tzigane parcmètre vodka."
def synthesize_sentence(sentence: str, output_sound):
	output_sound, sentence_data = synthesize_word(sentence, output_sound)
	return output_sound, sentence_data
concatenated_sound, sentence_data = synthesize_sentence(sentence, concatenated_sound)

# Snapshot the concatenation results before PSOLA
concatenated_sound.save(OUTPUT_WAV, "WAV")
print(concatenated_sound.n_samples)

# Compute PSOLA manipulations on the full conatenated sound, in order to have enough data to handle short phones.
# We'll just have to find our diphones positions again ;).
manip = pm.praat.call(concatenated_sound, "To Manipulation", 0.01, 75, 600)
pitch_tier = pm.praat.call(manip, "Extract pitch tier")
pm.praat.call(pitch_tier, "Remove points between", 0, concatenated_sound.duration)
duration_tier = pm.praat.call(manip, "Extract duration tier")

for phoneme_data in sentence_data:
	# In the concatenated stream
	start = phoneme_data["concat_start"]
	end = phoneme_data["concat_end"]
	mid = (start + end) / 2
	duration = phoneme_data["concat_duration"]
	print(f"concat duration: {duration}")
	# From espeak
	f0 = phoneme_data["f0"]
	target_duration = phoneme_data["duration"]
	print(f"target duration: {target_duration}")
	# We need an f0 ;)
	if f0 > 0:
		# Args: time, freq
		pm.praat.call(pitch_tier, "Add point", mid, f0)
	# No such restriction for duration
	# scale extracted phoneme to espeak phoneme's duration
	scale = target_duration / duration
	print(f"scaling point @ {mid} by {scale}")
	# Args: time, scale
	# FIXME: Do we need more points? Right now, Praat should lerp between points, which is probably good enough.
	# Another viable approach: 2 points, at start & end; in which case, a few ms away, to make the points unique between consecutive phonemes.
	pm.praat.call(duration_tier, "Add point", mid, scale)

# Apply the PSOLA manipulations
# NOTE: Since we apply everything at once, I assume this doesn't skew our timestamp positions given the duration changes?
pm.praat.call([manip, pitch_tier], "Replace pitch tier")
pm.praat.call([manip, duration_tier], "Replace duration tier")
modified_wav = pm.praat.call(manip, "Get resynthesis (overlap-add)")

# Format: also available via module constants, e.g., pm.SoundFileFormat.WAV
modified_wav.save(OUTPUT_FINAL_WAV, "WAV")
print(modified_wav.n_samples, modified_wav.get_total_duration(), format_duration(modified_wav.duration))
