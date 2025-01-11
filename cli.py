#!/usr/bin/env python3
#
# https://github.com/NiLuJe/TAL-S1-Synthese
#

import click
import synthese as Synthesize

# Quick'n dirty command via click
def set_param(ctx, param, value):
	"""Set a Synthesize settings"""

	if value is None or ctx.resilient_parsing:
		return
	name = param.human_readable_name.replace("-", "_")
	#print(f"Setting {name} to {value}")
	Synthesize.SETTINGS[name] = value

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(context_settings=CONTEXT_SETTINGS, epilog=Synthesize.print_available_sentences())
@click.option("-s", "--sentence",
				default=1,
				help="Which sentence to synthesize",
				type=click.IntRange(1, len(Synthesize.SENTENCES)),
				show_default=True)
@click.option("-v", "--voice",
				default=Synthesize.SETTINGS["voice"],
				help="eSpeak voice to use",
				type=click.Choice(Synthesize.list_espeak_voices()),
				callback=set_param,
				show_default=True)
@click.option("-g", "--word-gap",
				default=Synthesize.SETTINGS["word_gap"],
				help="eSpeak word gap, in seconds",
				callback=set_param,
				show_default=True)
@click.option("-p", "--pitch-multiplier",
				default=Synthesize.SETTINGS["pitch_multiplier"],
				help="eSpeak pitch multiplier",
				type=click.FloatRange(0.5, 2.0),
				callback=set_param,
				show_default=True)
@click.option("-r", "--pitch-range-multiplier",
				default=Synthesize.SETTINGS["pitch_range_multiplier"],
				help="eSpeak pitch range multiplier",
				type=click.FloatRange(0, 2.0),
				callback=set_param,
				show_default=True)
@click.option("-w", "--wpm",
				default=Synthesize.SETTINGS["wpm"],
				help="eSpeak words per minute",
				type=click.IntRange(80, 450),
				callback=set_param,
				show_default=True)
@click.option("-G", "--skip-word-gaps",
				default=Synthesize.SETTINGS["skip_word_gaps"],
				help="Do not honor eSpeak word-gap silences",
				callback=set_param,
				show_default=True)
@click.option("-D", "--duration-points",
				default=Synthesize.SETTINGS["duration_points"],
				help="How many duration points to use during PSOLA manipulations",
				type=click.Choice(["mid", "edges", "bracketed"]),
				callback=set_param,
				show_default=True)
@click.option("-P", "--pitch-points",
				default=Synthesize.SETTINGS["pitch_points"],
				help="How many pitch points to use during PSOLA manipulations",
				type=click.Choice(["mean", "trio"]),
				callback=set_param,
				show_default=True)
@click.option("-A", "--autoplay",
				default=Synthesize.SETTINGS["autoplay"],
				help="Play the final sound clip",
				callback=set_param,
				show_default=True)
@click.option("--verbose/--no-verbose",
				default=Synthesize.SETTINGS["verbose"],
				help="Print phoneme metadata during processing",
				callback=set_param,
				show_default=True)
@click.option("--debug/--no-debug",
				default=Synthesize.SETTINGS["debug"],
				help="Print even more phoneme metadata during processing",
				callback=set_param,
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
	autoplay: bool,
	verbose: bool,
	debug: bool
):
	"""CLI for Synthesize"""

	#print(Synthesize.SETTINGS)
	Synthesize.synthesize(Synthesize.SENTENCES[sentence-1])

# Main entry-point
if __name__ == "__main__":
	main(max_content_width=120)
