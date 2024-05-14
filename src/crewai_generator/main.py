#!/usr/bin/env python
from crewai_generator.crew import CrewaiGeneratorCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'AI LLMs'
    }
    CrewaiGeneratorCrew().crew().kickoff(inputs=inputs)