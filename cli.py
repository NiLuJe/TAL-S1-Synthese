#!/usr/bin/env python3
#
# https://github.com/NiLuJe/TAL-S1-Synthese
#

import click
import synthese as Synthesize

# Quick'n dirty command via click
@click.command()
@click.option("-s", "--sentence",				default=1,			help="Sentence to synthesize")
@click.option("-v", "--voice",					default="Male6",	help="eSpeak voice to use")
@click.option("-g", "--word-gap",				default=0.025,		help="eSpeak word gap, in seconds")
@click.option("-p", "--pitch-multiplier",		default=1.0,		help="eSpeak pitch multiplier")
@click.option("-r", "--pitch-range-multiplier",	default=1.0,		help="eSpeak pitch range multiplier")
@click.option("-w", "--wpm",					default=150,		help="eSpeak words per minute")
@click.option("-G", "--skip-word-gaps",			default=True,		help="Do not honor eSpeak word-gap silences")
@click.option("-D", "--duration-points",		default="mid",		help="How many duration points to use during PSOLA manipulations")
@click.option("-P", "--pitch-points",			default="mean",		help="How many pitch points to use during PSOLA manipulations")
@click.option("-A", "--autoplay",				default=False,		help="Play the final sound clip")

def main(
	sentence: str,
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
	"""CLI interface for Synthesize"""

	Synthesize.synthesize(Synthesize.SENTENCES[0])

if __name__ == '__main__':
	main()
