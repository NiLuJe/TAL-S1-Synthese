#!/usr/bin/env python3

import parselmouth as pm
import textgrids as tg
from pathlib import Path
import pprint

# Data types for typing annotations
from textgrids import Tier

# TODO: 1 vs. 3 point per phoneme contours, and keep it as an option
# TODO: Go through all the labels and build a phoneme bank in one go, then just query it.
# 		Make it a list, so we keep duplicates, and just choose one at random during synth.
# TODO: Make the voice choice an option (filter male voices?)
# TODO: Make the inter-word gap (if any?) configurable

# NOTE: Paths are relative to this file.
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / "data")
OUTPUT_DIR = (BASE_DIR / "output")
# NOTE: Unfortunately, parselmouth & co do not handle Path-like objects, so we resolve everything as strings...
SOUND_FILE = (DATA_DIR / "enregistrement.wav").as_posix()
GRID_FILE =  (DATA_DIR / "enregistrement.TextGrid").as_posix()
# Output files
OUTPUT_WAV = (OUTPUT_DIR / "raw_concat.wav").as_posix()
OUTPUT_SYNTHESIZED_WAV =  (OUTPUT_DIR / "espeak-synth.wav").as_posix()
OUTPUT_SYNTHESIZED_GRID = (OUTPUT_DIR / "espeak-synth.TextGrid").as_posix()
OUTPUT_FINAL_WAV =  (OUTPUT_DIR / "result.wav").as_posix()

# Sentence list
SENTENCES = [
	"Ah bah maintenant, elle va marcher beaucoup moins bien forcément !",
	"Je ne vous jette pas la pierre, Pierre, mais j'étais à deux doigts de m'agacer.",
	"Barrez-vous, cons de mimes !",
	"C'est une voiture de collection de prestige. Il y en a plus que trois qui roulent dans le monde et moi... J'ai la numéro 4."
]

# Utility functions
def format_duration(seconds: float) -> str:
	"""Returns a MM:SS.sss string for the given input float in seconds"""
	minutes = seconds // 60
	seconds, ms = divmod(seconds % 60, 1)
	# Round to 3 decimals, without the decimal point
	ms = str(round(ms, 3)).split(".")[1]
	return "{:02d}:{:02d}.{:s}".format(int(minutes), int(seconds), ms)

sound = pm.Sound(SOUND_FILE)
grid = tg.TextGrid(GRID_FILE)
pp = pm.praat.call(sound, "To PointProcess (zeroes)", 1, "yes", "no")
# NOTE: Layer name
diphones = grid["phone"]
# NOTE: Our grid was initially populated by Praat's Annotate > To TextGrd (silences) function.
#       At the time, we labeled silences with an empty label, and speech with an asterism.
#       Since we only want to match *consecutive* diphones, keeping those serves us well,
#       as it prevents us from skipping over labels when matching on a diphone,
#       which would risk mixing the wrong phones together...
#       This is especially important since we re-recorded a few logatomes missed during the initial recording *at the end* of the file...

# Extract a small slice of silence as our initial sound object.
concatenated_sound = sound.extract_part(0, 0.01, pm.WindowShape.RECTANGULAR, 1, False)
diphones_sound = {}

def extract_diphone(phoneme_1: str, phoneme_2: str, diphones: Tier):
	print(f"Extracting diphone {phoneme_1}{phoneme_2}...")
	for j, d in enumerate(diphones[:-1]):
		left = d
		right = diphones[j+1]

		phoneme = left.text
		next = right.text

		# For debugging purposes...
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
					"extracted_start": mid_left,
					"extracted_end": left.xmax,
					"extracted_duration": left.xmax - mid_left,	# i.e., duration / 2
				},
				{
					"phoneme": phoneme_2,
					"orig_start": right.xmin,
					"orig_end": right.xmax,
					"duration": right.dur,
					"diphone_pos": "right",
					"mid": mid_right,
					"extracted_start": right.xmin,
					"extracted_end": mid_right,
					"extracted_duration": mid_right - right.xmin, # i.e., also duration / 2
				}
			)
			# NOTE: Using non-rectangular window shapes (e.g., KAISER1) affects (i.e., attenuates) the edges too much for our use case, so stick to rectangular.
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

	# Strip the trailing _: (long pause on punctuation marks), _ (end of word pause) label pair, if any
	for i in range(n, 0, -1):
		label = pm.praat.call(text_synth, "Get label of interval", 4, i)
		if label == "_:":
			n = i-1
			break

	# Save the espeak synth for analysis & comparison
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

	# Compute the f0 mean of each phoneme (via Praat's "Get Mean")
	espeak_phonemes_start_ts = [pm.praat.call(text_synth, "Get start time of interval", 4, i + 1) for i in range(n)]
	espeak_phonemes_end_ts   = [pm.praat.call(text_synth, "Get end time of interval", 4, i + 1) for i in range(n)]

	# Make sure we start on silence, because apparently that's not a given (e.g., our first sentence)...
	if espeak_phonemes[0] != "_":
		espeak_phonemes.insert(0, "_")
		espeak_phonemes_start_ts.insert(0, 0.0)
		espeak_phonemes_end_ts.insert(0, 0.045)

	# Sanity check
	print(f"espeak transcription: {"".join(espeak_phonemes)}")

	# Zip it!
	espeak_data = []
	for i, (phoneme, start, end) in enumerate(zip(espeak_phonemes, espeak_phonemes_start_ts, espeak_phonemes_end_ts)):
		# FIXME: Skip pauses when they're not at the edges (can't add silence ths way... ;'()
		if 0 < i < len(espeak_phonemes)-1:
			if phoneme.startswith("_"):
				continue

		# NOTE: Kirshenbaum uses the IPA `ɡ` (U+0261), take care of it...
		if phoneme == "ɡ":
			# We prefer the ASCII `g` (U+0067)
			phoneme = "g"

		mean_f0 = pm.praat.call(pitch_synth, "Get mean", start, end, "Hertz")
		# We'll store everything in a list of dicts...
		d = {
			"phoneme": phoneme,
			"start": float(start),
			"end": float(end),
			"duration": float(end) - float(start),
			"f0": float(mean_f0),
		}
		espeak_data.append(d)

	# NOTE: Make sure we iterate on a list, and not a string, to handle diacritics properly...
	for i, label in enumerate(espeak_data[:-1]):
		phone1 = label["phoneme"]
		phone2 = espeak_data[i+1]["phoneme"]

		extraction, diphone_data = extract_diphone(phone1, phone2, diphones)

		# Concat
		if extraction != None and diphone_data != None:
			# Compute phoneme position in the concatenated stream, keeping in mind that two different diphones contribute to one phoneme...
			left_pos = output_sound.duration
			espeak_data[i]["concat_start"]    = espeak_data[i].get("concat_start", left_pos)
			espeak_data[i]["concat_duration"] = espeak_data[i].get("concat_duration", 0.0) + diphone_data[0]["extracted_duration"]
			espeak_data[i]["concat_end"]      = espeak_data[i]["concat_start"] + espeak_data[i]["concat_duration"]

			right_pos = espeak_data[i]["concat_end"]
			espeak_data[i+1]["concat_start"]    = espeak_data[i+1].get("concat_start", right_pos)
			espeak_data[i+1]["concat_duration"] = espeak_data[i+1].get("concat_duration", 0.0) + diphone_data[1]["extracted_duration"]
			espeak_data[i+1]["concat_end"]      = espeak_data[i+1]["concat_start"] + espeak_data[i+1]["concat_duration"]
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
			print(f"Failed to extract diphone {phone1}{phone2}")
	return (output_sound, espeak_data)

# FIXME: Or sys.argv[1]
sentence = SENTENCES[0]

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
	print(f"phoneme: {phoneme_data["phoneme"]}")
	# In the concatenated stream
	start = phoneme_data["concat_start"]
	end = phoneme_data["concat_end"]
	mid = (start + end) / 2
	duration = phoneme_data["concat_duration"]
	print(f"concat duration: {duration} ({start} -> {end})")
	# From espeak
	f0 = phoneme_data["f0"]
	target_duration = phoneme_data["duration"]
	print(f"target duration: {target_duration} ({phoneme_data["start"]} -> {phoneme_data["end"]})")
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
	pm.praat.call(duration_tier, "Add point", mid, scale)

	# Another viable approach: 2 points, at start & end; in which case, a few ms away, to make the points unique between consecutive phonemes.
	# NOTE: This matches how the Praat manual explains that sort of stuff...
	# The best approach miight depend on the voice, or the "shape" of the scaling, basically?
	#pm.praat.call(duration_tier, "Add point", start + 0.001, scale)
	#pm.praat.call(duration_tier, "Add point", end - 0.001, scale)

	#pm.praat.call(duration_tier, "Add point", start + 0.001, 1)
	#pm.praat.call(duration_tier, "Add point", end - 0.001, 1)
	#pm.praat.call(duration_tier, "Add point", start + 0.002, scale)
	#pm.praat.call(duration_tier, "Add point", end - 0.002, scale)

# Apply the PSOLA manipulations
# NOTE: Since we apply everything at once, I assume this doesn't skew our timestamp positions given the duration changes?
pm.praat.call([manip, pitch_tier], "Replace pitch tier")
pm.praat.call([manip, duration_tier], "Replace duration tier")
modified_wav = pm.praat.call(manip, "Get resynthesis (overlap-add)")

# Format: also available via module constants, e.g., pm.SoundFileFormat.WAV
modified_wav.save(OUTPUT_FINAL_WAV, "WAV")
print(modified_wav.n_samples, modified_wav.get_total_duration(), format_duration(modified_wav.duration))
