#!/usr/bin/env python3

# Whee, colors!
from colorama import just_fix_windows_console
# Make the Windows terminal handle ANSI escape sequences sanely...
just_fix_windows_console()
from rich import print
from rich.pretty import pprint

import itertools
import math
import parselmouth as pm
import textgrids as tg
from pathlib import Path

# Data types for typing annotations
from typing import Any
from textgrids import Tier
from parselmouth import Sound, Data

# TODO: Go through all the labels and build a diphone bank in one go, then just query it.
# 		Make it a list, so we keep duplicates, and just choose one at random during synth.
#       Also remember the original position, and default to choosing the closest pos to the prev match (i.e., in order, make it the default).

# NOTE: Paths are relative to this file.
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / "data")
OUTPUT_DIR = (BASE_DIR / "output")
# NOTE: Unfortunately, parselmouth & co do not handle Path-like objects, so we resolve everything as strings...
SOUND_FILE = (DATA_DIR / "enregistrement.wav").as_posix()
GRID_FILE =  (DATA_DIR / "enregistrement.TextGrid").as_posix()
# Output files
CONCAT_WAV = (OUTPUT_DIR / "raw_concat.wav").as_posix()
ESPEAK_WAV =  (OUTPUT_DIR / "espeak-synth.wav").as_posix()
ESPEAK_GRID = (OUTPUT_DIR / "espeak-synth.TextGrid").as_posix()
OUTPUT_FINAL_WAV =  (OUTPUT_DIR / "result.wav").as_posix()

# Sentence list
SENTENCES = [
	"Ah bah maintenant, elle va marcher beaucoup moins bien forcément !",
	"Je ne vous jette pas la pierre, Pierre, mais j'étais à deux doigts de m'agacer.",
	"Barrez-vous, cons de mimes !",
	"C'est une voiture de collection de prestige. Il y en a plus que trois qui roulent dans le monde et moi... J'ai la numéro 4."
]

# Settings
SETTINGS = {
	# For eSpeak
	"voice": "Male6", # m6 seems to match the default for roa/fr in espeak-ng...
	"word_gap": 0.025, # in seconds
	"pitch_multiplier": 1.0, # 0.5-2.0
	"pitch_range_multiplier": 1.0, # 0-2.0
	"wpm": 150, # 80-450, Praat's default is 175
	# Behavior tweaks
	"skip_word_gaps": False, # Add word_gap silences on espeak word gaps if False, otherwise, skip them
	"duration_points": "mid", # How many duration points to use during PSOLA (mid: a single point at the midpoint of the phone; edges: two points at the edges of the phoneme, bracketed: edges, bracketed by neutral points)
	"pitch_points": "mean", # How many pitch points to copy from eSpeak (mean: a single point, set to the mean; trio: three points: start, mid, end)
}

# Utility functions
def format_duration(seconds: float) -> str:
	"""Returns a MM:SS.sss string for the given input float in seconds"""

	minutes = seconds // 60
	seconds, ms = divmod(seconds % 60, 1)
	# Round to 3 decimals, without the decimal point
	ms = str(round(ms, 3)).split(".")[1]
	return "{:02d}:{:02d}.{:03s}".format(int(minutes), int(seconds), ms)

def print_sound_info(sound: Sound):
	"""Print detailed information about a Sound object"""

	print(sound.info(), end="")
	print(f"Duration: {format_duration(sound.duration)}")


# Global objects
DIPHONES_SOUND = pm.Sound(SOUND_FILE)
DIPHONES_GRID = tg.TextGrid(GRID_FILE)
DIPHONES_PP = pm.praat.call(DIPHONES_SOUND, "To PointProcess (zeroes)", 1, "yes", "no")
# NOTE: Layer name
DIPHONES_TIER = DIPHONES_GRID["phone"]
# NOTE: Our grid was initially populated by Praat's Annotate > To TextGrid (silences) function.
#       At the time, we labeled silences with an empty label, and speech with an asterism.
#       Since we only want to match *consecutive* diphones, keeping those serves us well,
#       as it prevents us from skipping over labels when matching on a diphone,
#       which would risk mixing the wrong phones together...
#       This is especially important since we re-recorded a few logatomes missed during the initial recording *at the end* of the file...

# Generate a small slice of silence as our initial sound object
# Args: obj name, end, start, duration, samplerate, formula
CONCAT_SOUND = pm.praat.call("Create Sound from formula", "concat", 1, 0, SETTINGS["word_gap"], 16000, str(0))


def extract_diphone(phoneme_1: str, phoneme_2: str, sound: Sound, diphones: Tier, pp: Data) -> tuple[Sound | None, tuple[dict[str, Any], dict[str, Any]] | None]:
	"""
	Extract the diphone `phoneme_1` + `phoneme_2` from the Sound `sound` via the Tier `diphones` & PointProcess `pp`.
	Returns a 2-element tuple composed of a Sound object, and a tuple of 2 dictionaries with metadata for the individual phones.
	Returns (None, None) on extraction failure.
	"""

	print(f"Extracting diphone {phoneme_1}{phoneme_2}...")
	# NOTE: pairwise does exactly what we need, iterating over overlapping pairs ;).
	for pair in itertools.pairwise(diphones):
		left  = pair[0]
		right = pair[1]

		phoneme = left.text
		next = right.text

		# For debugging purposes...
		#print(f"{phoneme} vs. {phoneme_1} // {next} vs. {phoneme_2}")
		if phoneme == phoneme_1 and next == phoneme_2:
			mid_left = left.mid
			mid_right = right.mid

			# NOTE: We can't choose only rising or descending crossings...
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
			# NOTE: We store more data than striclty needed, to ease human verification ;).
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

def find_pitch_point(pitch_obj: Data, start: float, end: float, which: str) -> float:
	"""Scour a Pitch object `pitch_object` for a pitch point between positions `start` and `end`, looking for the start/mid/end points."""

	mid = (start + end) / 2
	f0 = float("nan")
	offset = 0.0

	match which:
		case "start":
			while math.isnan(f0):
				pos = start + offset
				f0 = pm.praat.call(pitch_obj, "Get value at time", pos, "Hertz", "linear")
				#print("start_f0", f0, offset)

				# Look 1ms away next...
				offset += 0.001
				# Don't go OOB of the phoneme
				if pos >= end:
					break
		case "end":
			while math.isnan(f0):
				pos = end - offset
				f0 = pm.praat.call(pitch_obj, "Get value at time", pos, "Hertz", "linear")
				#print("end_f0", f0, offset)

				offset += 0.001
				if pos <= start:
					break
		case "mid":
			while math.isnan(f0):
				# Look ahead...
				pos = mid + offset
				f0 = pm.praat.call(pitch_obj, "Get value at time", pos, "Hertz", "linear")
				#print("(ahead) mid_f0", f0, offset)

				# Did we get it?
				if not math.isnan(f0):
					break

				# Nope, look behind...
				pos = mid - offset
				f0 = pm.praat.call(pitch_obj, "Get value at time", pos, "Hertz", "linear")
				#print("(behind) mid_f0", f0, offset)

				offset += 0.001
				if pos <= start or pos >= end:
					break

	return f0

def espeak_sentence(sentence: str, output_sound_path: str, output_grid_path: str) -> list[dict[str, Any]]:
	"""
	Synthesize sentence `sentence` via Praat's eSpeak implementation.
	Save the results (sound & grid) to `output_sound_path` & `output_grid_path`, respectively.
	Returns a list of dictionaries with metadata about each phoneme generated,
	to be used for PSOLA manipulations later on.
	"""

	print(f"espeak_sentence: {sentence}")
	praat_synth = pm.praat.call("Create SpeechSynthesizer", "French (France)", SETTINGS["voice"])
	# Setup espeak to use XSampa, and honor our settings
	pm.praat.call(praat_synth,
				  "Speech output settings",
				  16000,
				  SETTINGS["word_gap"],
				  SETTINGS["pitch_multiplier"],
				  SETTINGS["pitch_range_multiplier"],
				  SETTINGS["wpm"],
				  "Kirshenbaum_espeak")
	text_synth, sound_synth = pm.praat.call(praat_synth, "To Sound", sentence, "yes")
	n = pm.praat.call(text_synth, "Get number of intervals", 4)

	# Strip the trailing _: (long pause on punctuation marks), _ (end of word pause) label pair, if any
	for i in range(n, 0, -1):
		label = pm.praat.call(text_synth, "Get label of interval", 4, i)
		if label == "_:":
			n = i-1
			break

	# Save the espeak synth for analysis & comparison
	text_synth.save(output_grid_path)
	# Format: also available via module constants, e.g., pm.SoundFileFormat.WAV
	sound_synth.save(output_sound_path, "WAV")

	# I was getting some pretty weird outliers w/ shs when slowing down eSpeak's wpm (on voiceless phonemes, even!)...
	#pitch_synth = pm.praat.call(sound_synth, "To Pitch (shs)", 0.01, 50, 15, 1250, 15, 0.84, 600, 48)
	# NOTE: Praat's docs recommend this algo for intonation, but Parselmouh 0.4.5 ships with an older internal Praat copy...
	#pitch_synth = pm.praat.call(sound_synth, "To Pitch (filtered autocorrelation)", 0, 50, 800, 15, "yes", 0.03, 0.09, 0.5, 0.055, 0.35, 0.14)
	# We can use raw ac instead, but under its old name, which takes arguments in a slightly different order...
	# c.f., https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__ac____.html
	#pitch_synth = pm.praat.call(sound_synth, "To Pitch (raw autocorrelation)", 0, 75, 600, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14)
	pitch_synth = pm.praat.call(sound_synth, "To Pitch (ac)", 0.0, 75, 15, "yes", 0.03, 0.45, 0.01, 0.35, 0.14, 600)

	# NOTE: This includes word-gaps, and silences on punctuation marks.
	#       See the whole skip_word_gaps codepaths...
	espeak_phonemes = [pm.praat.call(text_synth, "Get label of interval", 4, i + 1) for i in range(n)]
	print(f"espeak_phonemes: {espeak_phonemes}")
	# Replace empty phonemes w/ an underscore
	espeak_phonemes = ["_" if x == "" else x for x in espeak_phonemes]

	# We'll need those to compute the f0 mean of each phoneme (via Praat's "Get Mean")
	espeak_phonemes_start_ts = [pm.praat.call(text_synth, "Get start time of interval", 4, i + 1) for i in range(n)]
	espeak_phonemes_end_ts   = [pm.praat.call(text_synth, "Get end time of interval", 4, i + 1) for i in range(n)]

	# Make sure we start on silence, because apparently that's not a given (e.g., our first sentence)...
	if espeak_phonemes[0] != "_":
		espeak_phonemes.insert(0, "_")
		espeak_phonemes_start_ts.insert(0, 0.0)
		espeak_phonemes_end_ts.insert(0, SETTINGS["word_gap"] * 2)

	# Sanity check
	print(f"espeak transcription: {"".join(espeak_phonemes)}")

	# Zip it all together!
	espeak_data = []
	for i, (phoneme, start, end) in enumerate(zip(espeak_phonemes, espeak_phonemes_start_ts, espeak_phonemes_end_ts)):
		# NOTE: When we don't want to insert silences at word-gaps, just skip them entirely.
		#       This makes the following loops slightly saner to follow,
		#       (c.f., before f89c1264f980c615a3c41ca0c998cb4f5d9cf8a1).
		if SETTINGS["skip_word_gaps"]:
			if 0 < i < len(espeak_phonemes)-1:
				if phoneme.startswith("_"):
					continue

		# NOTE: Kirshenbaum uses the IPA `ɡ` (U+0261), take care of it...
		if phoneme == "ɡ":
			# We prefer the ASCII `g` (U+0067)
			phoneme = "g"

		# We default to the mean pitch...
		mean_f0 = pm.praat.call(pitch_synth, "Get mean", start, end, "Hertz")
		#print(phoneme, mean_f0, start, end)

		# But also store a few pitch points to experiment with, if requested...
		mid = (float(start) + float(end)) / 2
		start_f0 = float("nan")
		mid_f0 = float("nan")
		end_f0 = float("nan")
		# Don't compute them if we won't use them
		if SETTINGS["pitch_points"] == "trio":
			# NOTE: This is trickier than the mean, because there may be undefined pitch points (or none at all, for voiceless phonemes)...
			#       See the `find_pitch_point` implementation for more details.
			start_f0 = find_pitch_point(pitch_synth, start, end, "start")
			mid_f0   = find_pitch_point(pitch_synth, start, end, "mid")
			end_f0   = find_pitch_point(pitch_synth, start, end, "end")

		# We'll store everything in a list of dicts...
		d = {
			"idx": len(espeak_data), # We'll need to poke at the actual data inside a pairwise iterator, so remember the index
			"phoneme": phoneme,
			"start": float(start),
			"end": float(end),
			"mid": mid,
			"duration": float(end) - float(start),
			"f0": float(mean_f0),
			"start_f0": start_f0,
			"mid_f0": mid_f0,
			"end_f0": end_f0,
		}
		espeak_data.append(d)
	return espeak_data

def synthesize_sentence(sentence: str, output_sound: Sound) -> tuple[Sound, list[dict[str, Any]]]:
	"""
	Synthesize sentence `sentence`, and concatenate the results in `output_sound`.
	Returns a tuple with said sound object, and a list of metadata dictionaries for each phoneme,
	like `espeak_sentence`.
	"""

	# Let eSpeak do its thing first
	espeak_data = espeak_sentence(sentence, ESPEAK_WAV, ESPEAK_GRID)

	real_left, real_right = None, None
	# NOTE: Make sure we iterate on a list, and not a string, to handle diacritics properly...
	for i, pair in enumerate(itertools.pairwise(espeak_data)):
		# Trickery needed to deal with word-gaps...
		left  = real_left or pair[0]
		right = real_right or pair[1]

		left_i = left["idx"]
		phone1 = left["phoneme"]
		right_i = right["idx"]
		phone2 = right["phoneme"]
		print(f"Iterating on diphone: {phone1}{phone2}")

		# NOTE: Handle word gaps manually, as we only annotate "long" diphones on *sentence* edges..
		if 0 < left_i < len(espeak_data)-2:
			# NOTE: We drop them entirely from espeak_data w/ skip_word_gaps, so no need to re-check that setting here
			if phone1.startswith("_") or phone2.startswith("_"):
				# Create a chunk of silence
				duration = SETTINGS["word_gap"] * 4 if phone1.endswith(":") or phone2.endswith(":") else SETTINGS["word_gap"]
				print(f"Inserting a {duration} seconds word gap silence")
				silence = pm.praat.call("Create Sound from formula", "silence", 1, 0, duration, 16000, str(0))
				# Insert it w/o metadata, we won't need it for PSOLA later
				output_sound = output_sound.concatenate([output_sound, silence])
				# Remember the proper phoneme to use for the next iteration (i.e., skip over the silences while keeping track of the previous phone)...
				if phone2.startswith("_"):
					real_left = espeak_data[left_i]
				else:
					real_right = espeak_data[right_i]
				# And skip right to the next iteration
				continue

		extraction, diphone_data = extract_diphone(phone1, phone2, DIPHONES_SOUND, DIPHONES_TIER, DIPHONES_PP)

		# Concat
		if extraction != None and diphone_data != None:
			# Compute phoneme position in the concatenated stream, keeping in mind that two different diphones contribute to one phoneme...
			#print("starting espeak_data (left):")
			#pprint(espeak_data[left_i], expand_all=True)
			left_pos = output_sound.duration
			espeak_data[left_i]["concat_start"]    = espeak_data[left_i].get("concat_start", left_pos)
			espeak_data[left_i]["concat_duration"] = espeak_data[left_i].get("concat_duration", 0.0) + diphone_data[0]["extracted_duration"]
			espeak_data[left_i]["concat_end"]      = espeak_data[left_i]["concat_start"] + espeak_data[left_i]["concat_duration"]

			#print("starting espeak_data (right):")
			#pprint(espeak_data[right_i], expand_all=True)
			# NOTE: We can't use espeak_data[left_i]["concat_end"] because that would fail to account for inserted word-gap silences
			right_pos = output_sound.duration + diphone_data[0]["extracted_duration"]
			espeak_data[right_i]["concat_start"]    = espeak_data[right_i].get("concat_start", right_pos)
			espeak_data[right_i]["concat_duration"] = espeak_data[right_i].get("concat_duration", 0.0) + diphone_data[1]["extracted_duration"]
			espeak_data[right_i]["concat_end"]      = espeak_data[right_i]["concat_start"] + espeak_data[right_i]["concat_duration"]
			# Sanity check...
			# NOTE:
			#	- In diphone_data:
			#		- orig ts are ts in the original recording
			#	- In espeak data:
			#		- unprefixed ts are ts in the espeak synth
			#		- concat ts are the output ts in the concatenated stream
			print("diphone_data (left):")
			pprint(diphone_data[0], expand_all=True)
			print("espeak_data (left):")
			pprint(espeak_data[left_i], expand_all=True)
			print("diphone_data (right):")
			pprint(diphone_data[1], expand_all=True)
			print("espeak_data (right):")
			pprint(espeak_data[right_i], expand_all=True)

			output_sound = output_sound.concatenate([output_sound, extraction])
		else:
			print(f"[bold red]!! Failed to extract diphone[/bold red] [bold green]{phone1}{phone2}[/bold green]")
		# That was a real dihone extraction, clear the word gap tracking...
		real_left, real_right = None, None
	return (output_sound, espeak_data)

def manipulate_sound(concatenated_sound: Sound, sentence_data: list[dict[str, Any]]) -> Sound:
	"""
	Iterate over `concatenated_sound` phoneme by phoneme, applying the f0 contour (if any) and duration of the matching phoneme from an espeak synth,
	via the metadata available in `sentence_data` (as produced by `espeak_sentence` & `synthesize_sentence`).
	Returns a new Sound object with the manipulations applied via PSOLA.
	"""

	# Compute PSOLA manipulations on the full concatenated sound, in order to have enough data to handle short phones.
	# We'll just have to find our phoneme positions again, hence the metadata in `sentence_data` ;).
	manip = pm.praat.call(concatenated_sound, "To Manipulation", 0.01, 75, 600)
	pitch_tier = pm.praat.call(manip, "Extract pitch tier")
	pm.praat.call(pitch_tier, "Remove points between", 0, concatenated_sound.duration)
	duration_tier = pm.praat.call(manip, "Extract duration tier")

	# Iterate phoneme by phoneme over their metadata
	for i, phoneme_data in enumerate(sentence_data):
			phoneme = phoneme_data["phoneme"]

			# NOTE: Leave inter-word gaps we handled as silences alone
			if 0 < i < len(sentence_data)-1:
				if phoneme.startswith("_"):
					continue

			print(f"phoneme: {phoneme}")
			# In the concatenated stream
			start = phoneme_data["concat_start"]
			end = phoneme_data["concat_end"]
			mid = (start + end) / 2
			duration = phoneme_data["concat_duration"]
			print(f"concat duration: {duration} ({start} -> {end})")
			# From eSpeak
			target_duration = phoneme_data["duration"]
			print(f"target duration: {target_duration} ({phoneme_data["start"]} -> {phoneme_data["end"]})")

			# We can experiment with a couple variations on how to apply the pitch...
			match SETTINGS["pitch_points"]:
				case "mean":
					# We need an f0 ;)
					mean_f0 = phoneme_data["f0"]
					if mean_f0 > 0:
						print(f"Adding a mean pitch point of {mean_f0} Hz @ midpoint {mid}")
						# Args: time, freq
						pm.praat.call(pitch_tier, "Add point", mid, mean_f0)
					else:
						print("[yellow]No pitch data[/yellow]")
				case "trio":
					start_f0 = phoneme_data["start_f0"]
					mid_f0   = phoneme_data["mid_f0"]
					end_f0   = phoneme_data["end_f0"]
					if start_f0 > 0 and mid_f0 > 0 and end_f0 > 0:
						print(f"Adding three pitch points [{start_f0}, {mid_f0}, {end_f0}] Hz @ [{start}, {mid}, {end}]")
						# NOTE: We plop these right at the edges, while we may have captured them slightly further away than that during the search...
						pm.praat.call(pitch_tier, "Add point", start + 0.001, start_f0)
						pm.praat.call(pitch_tier, "Add point", mid, mid_f0)
						pm.praat.call(pitch_tier, "Add point", end - 0.001, end_f0)
					else:
						print("[yellow]No pitch data[/yellow]")
				case _:
					print(f"[red]!! Invalid `pitch_points` setting:[/red] [green]{SETTINGS["pitch_points"]}[/green]")

			# No need to validate target_duration, on the other hand, it's guaranteed to be non-zero.
			# scale extracted phoneme to eSpeak phoneme's duration
			scale = target_duration / duration
			# We can experiment with a few variations on how to apply the duration...
			match SETTINGS["duration_points"]:
				case "mid":
					print(f"scaling midpoint @ {mid} by {scale}")
					# Args: time, scale
					pm.praat.call(duration_tier, "Add point", mid, scale)
				case "edges":
					print(f"scaling from {start} to {end} by {scale}")
					# NOTE: This matches how the Praat manual explains that sort of stuff...
					# The best approach miiiight actually depend on the voice, or the "shape" of the scaling, basically?
					pm.praat.call(duration_tier, "Add point", start + 0.001, scale)
					pm.praat.call(duration_tier, "Add point", end - 0.001, scale)
				case "bracketed":
					print(f"bracketed scaling from {start} to {end} by {scale}")
					pm.praat.call(duration_tier, "Add point", start + 0.001, 1)
					pm.praat.call(duration_tier, "Add point", end - 0.001, 1)
					pm.praat.call(duration_tier, "Add point", start + 0.002, scale)
					pm.praat.call(duration_tier, "Add point", end - 0.002, scale)
				case _:
					print(f"[red]!! Invalid `duration_points` setting:[/red] [green]{SETTINGS["duration_points"]}[/green]")

	# Apply the PSOLA manipulations
	# NOTE: Since we apply everything at once, I assume this doesn't skew our timestamp positions given the duration changes?
	#       AFAICT, there's a TextGrid time scaling function to deal with this when you need to preserve a TextGrid.
	#       We don't, so we're good :).
	pm.praat.call([manip, pitch_tier], "Replace pitch tier")
	pm.praat.call([manip, duration_tier], "Replace duration tier")
	return pm.praat.call(manip, "Get resynthesis (overlap-add)")

def synthesize(sentence: str):
	output_sound, sentence_data = synthesize_sentence(sentence, CONCAT_SOUND)

	# Snapshot the concatenation results before PSOLA
	output_sound.save(CONCAT_WAV, "WAV")
	print_sound_info(output_sound)

	# Apply PSOLA manipulations
	modified_wav = manipulate_sound(output_sound, sentence_data)

	# And, finally, save the final result!
	modified_wav.save(OUTPUT_FINAL_WAV, "WAV")
	print_sound_info(modified_wav)

# Main entry-point
if __name__ == "__main__":
	# Throw the first sentence at it for a quick sanity check
	synthesize(SENTENCES[0])
