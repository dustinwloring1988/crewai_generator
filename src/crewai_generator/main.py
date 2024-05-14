#!/usr/bin/env python
from crewai_generator.crew import CrewaiGeneratorCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'library_name': 'https://github.com/joaomdmoura/crewAI-tools/blob/main/crewai_tools/tools/code_docs_search_tool/README.md'
    }
    CrewaiGeneratorCrew().crew().kickoff(inputs=inputs)
