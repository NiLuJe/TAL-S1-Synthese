#!/usr/bin/env python3
#
# https://github.com/NiLuJe/TAL-S1-Synthese
#

import click
import synthese as Synthesize

# Quick'n dirty command via click
@click.command()
@click.option("-s", "--sentence",
				default=1,
				help="Sentence number to synthesize",
				type=click.IntRange(1, len(Synthesize.SENTENCES)),
				show_default=True)
@click.option("-v", "--voice",
				default=Synthesize.SETTINGS["voice"],
				help="eSpeak voice to use",
				show_default=True)
@click.option("-g", "--word-gap",
				default=Synthesize.SETTINGS["word_gap"],
				help="eSpeak word gap, in seconds",
				show_default=True)
@click.option("-p", "--pitch-multiplier",
				default=Synthesize.SETTINGS["pitch_multiplier"],
				help="eSpeak pitch multiplier",
				type=click.FloatRange(0.5, 2.0),
				show_default=True)
@click.option("-r", "--pitch-range-multiplier",
				default=Synthesize.SETTINGS["pitch_range_multiplier"],
				help="eSpeak pitch range multiplier",
				type=click.FloatRange(0, 2.0),
				show_default=True)
@click.option("-w", "--wpm",
				default=Synthesize.SETTINGS["wpm"],
				help="eSpeak words per minute",
				type=click.IntRange(80, 450),
				show_default=True)
@click.option("-G", "--skip-word-gaps",
				default=Synthesize.SETTINGS["skip_word_gaps"],
				help="Do not honor eSpeak word-gap silences",
				show_default=True)
@click.option("-D", "--duration-points",
				default=Synthesize.SETTINGS["duration_points"],
				help="How many duration points to use during PSOLA manipulations",
				type=click.Choice(["mid", "edges", "bracketed"]),
				show_default=True)
@click.option("-P", "--pitch-points",
				default=Synthesize.SETTINGS["pitch_points"],
				help="How many pitch points to use during PSOLA manipulations",
				type=click.Choice(["mean", "trio"]),
				show_default=True)
@click.option("-A", "--autoplay",
				default=Synthesize.SETTINGS["autoplay"],
				help="Play the final sound clip",
				show_default=True)

def main(
	sentence: int,
	voice: str,
	word_gap: float,
	pitch_multiplier: float,
	pitch_range_multiplier: float,
	wpm: int,
	skip_word_gaps: bool,
	duration_points: str,
	pitch_points: str,
	autoplay: bool
):
	"""CLI for Synthesize"""

	Synthesize.synthesize(Synthesize.SENTENCES[sentence-1])

if __name__ == '__main__':
	main()
