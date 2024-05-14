from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

from crewai_tools import (
    ScrapeWebsiteTool,
    ScrapeElementFromWebsiteTool,
    RagTool  # Importing the general-purpose RAG tool
)

agentops.init()

@CrewBase
class CrewaiGeneratorCrew():
	"""CrewaiGenerator crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        """
        Initializes the object with a ChatGroq instance for Groq operations.

        Parameters:
            self: The object instance.
        
        Returns:
            None
        """
        # Groq
        self.groq_llm = ChatGroq(
            temperature=0,
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
        )

    @agent
    def web_scraper_agent(self) -> Agent:
        """
        Defines the web scraping agent function that returns an Agent object.

        Parameters:
            self: The object instance.

        Returns:
            Agent: An instance of the Agent class.
        """
        return Agent(
            config=self.agents_config['web_scraper_agent'],  # Adjusted agent configuration
            tools=[ScrapeWebsiteTool(), ScrapeElementFromWebsiteTool(), RagTool()],  # Included necessary tools
            llm=self.groq_llm,  
            verbose=True
        )
		
    @agent
    def data_cleaner_agent(self) -> Agent:
        """
        Returns an Agent object based on the data_cleaner_agent configuration and verbosity setting.
        """
        return Agent(
            config=self.agents_config['data_cleaner_agent'],  # Adjusted agent configuration
            tools=[RagTool()],  # Included necessary tools
            verbose=True
        )
		
    @agent
    def question_generation_agent(self) -> Agent:
        """
        Returns an Agent object based on the question_generation_agent configuration and verbosity setting.
        """
        return Agent(
            config=self.agents_config['question_generation_agent'],  # Adjusted agent configuration
            tools=[RagTool()],  # Included necessary tools
            verbose=True
        )

    @agent
    def quality_control_specialist(self) -> Agent:
        """
        Returns an Agent object based on the quality_control_specialist configuration and verbosity setting.
        """
        return Agent(
            config=self.agents_config['quality_control_specialist'],  # Adjusted agent configuration
            tools=[RagTool()],  # Included necessary tools
            verbose=True
        )

    @task
    def extract_library_features_task(self) -> Task:
        """
        Extracts library features based on the configuration and the web scraping agent.
        """
        return Task(
            config=self.tasks_config['extract_library_features_task'],  # Adjusted task configuration
            agent=self.web_scraper_agent()  # Using the modified web_scraper_agent
        )

    @task
    def generate_QA_pairs_task(self) -> Task:
        """
        Generates question and answer pairs using the cleaned data and the question generation agent.
        """
        return Task(
            config=self.tasks_config['generate_QA_pairs_task'],  # Adjusted task configuration
            agent=self.question_generation_agent()  # Using the modified question_generation_agent
        )

    @task
    def analyze_documentation_insights_task(self) -> Task:
        """
        Analyzes documentation insights based on the provided configuration and the web scraping agent.
        """
        return Task(
            config=self.tasks_config['analyze_documentation_insights_task'],  # Adjusted task configuration
            agent=self.web_scraper_agent(),  # Using the modified web_scraper_agent
            output_file='documentation_insights_report.md'
        )

    @task
    def dataset_quality_control_task(self) -> Task:
        """
        Conducts quality control on the generated question and answer pairs dataset.
        """
        return Task(
            config=self.tasks_config['dataset_quality_control_task'],  # Adjusted task configuration
            agent=self.quality_control_specialist()  # Using the modified quality_control_specialist
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Python Library Dataset Generation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,
            max_rpm=3,
            max_iter=2,
            verbose=2,
        )
