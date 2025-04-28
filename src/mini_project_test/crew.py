from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
import os
from tool.weather_tool import WeatherTool
from crewai_tools import SerperDevTool

# import litellm
# litellm._turn_on_debug()

# Initialize the tool for internet searching capabilities
tool = SerperDevTool()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
crop_pdf_source = PDFKnowledgeSource(
    file_paths=["crop/millet.pdf", "crop/oilseeds.pdf", "crop/pulses.pdf", "crop/rice.pdf", "crop/sugarcrop.pdf", "agro-climatic-zone/agroclimatic-region.pdf", "season/Planting-Seasons-in-Tamil-Nadu.pdf"]
)

soil_pdf_source = PDFKnowledgeSource(
    file_paths=["soil/soil-type.pdf"]
)

fertilizer_pdf_source = PDFKnowledgeSource(
    file_paths=["primary-concern/nutrient-management.pdf"]
)

schemes_pdf_source = PDFKnowledgeSource(
    file_paths=["schemes/schemes.pdf"]
)

@CrewBase
class MiniProjectTest():
	"""MiniProjectTest crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	gemini_llm = LLM(
		model="gemini/gemini-2.0-flash",
		api_key=GEMINI_API_KEY,
		temperature=0,
	)
	ollama_llm = LLM(
		model = 'ollama/hf.co/mradermacher/dhenu2-in-climate-llama3.2-1b-i1-GGUF:Q4_K_M',
		base_url='http://localhost:11434',
	)
	tamil_llm = LLM(
		model = 'ollama/hf.co/mradermacher/dhenu2-in-climate-llama3.2-1b-i1-GGUF:Q4_K_M',
		base_url='http://localhost:11434',
	)


	# ollama_gemini_llm= LLM(
	# 	model = 'ollama/hf.co/mradermacher/oh-dcft-v3.1-gemini-1.5-pro-i1-GGUF:Q4_K_M',
	# 	base_url='http://localhost:11434',
	# )
	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def crop_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['crop_agent'],
			verbose=True,
			knowledge_sources=[crop_pdf_source],
			# llm = self.ollama_llm,
			llm = self.gemini_llm,
			# llm = self.ollama_gemini_llm,
			# embedder={
			# 	"provider": "ollama",
			# 	"config": {
			# 		"model": "nomic-embed-text",
			# 		"base_url": "http://localhost:11434"
			# 	}
        	# },
			embedder={
        		"provider": "google",
        		"config": {
            		"model": "models/text-embedding-004",
            		"api_key": GEMINI_API_KEY,
        		}
    		},
		)
	

	@agent
	def climate_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['climate_agent'],
			verbose=True,
			tools=[WeatherTool()],
			# llm = self.gemini_llm,
			# llm = self.ollama_gemini_llm
			# llm = self.ollama_llm
		)
	
	
	@agent
	def soil_health_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['soil_health_agent'],
			verbose=True,
			# llm = self.ollama_llm,
			knowledge_sources=[soil_pdf_source],
			# llm = self.ollama_llm,
			llm = self.gemini_llm,
			# llm = self.ollama_gemini_llm,
			# embedder={
			# 	"provider": "ollama",
			# 	"config": {
			# 		"model": "nomic-embed-text",
			# 		"base_url": "http://localhost:11434"
			# 	}
        	# },
			embedder={
        		"provider": "google",
        		"config": {
            		"model": "models/text-embedding-004",
            		"api_key": GEMINI_API_KEY,
        		}
    		},
		)

	@agent
	def fertilizer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['fertilizer_agent'],
			verbose=True,
			knowledge_sources=[crop_pdf_source, soil_pdf_source],
			# llm = self.gemini_llm,
			llm = self.ollama_llm,
			# llm = self.ollama_gemini_llm,
			# embedder={
			# 	"provider": "ollama",
			# 	"config": {
			# 		"model": "nomic-embed-text",
			# 		"base_url": "http://localhost:11434"
			# 	}
			# },
			embedder={
        		"provider": "google",
        		"config": {
            		"model": "models/text-embedding-004",
            		"api_key": GEMINI_API_KEY,
        		}
    		},
		)

	@agent
	def scheme_policy_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['scheme_policy_agent'],
			verbose=True,
			knowledge_sources=[schemes_pdf_source],
			llm = self.ollama_llm,
			# llm= self.gemini_llm,
			# llm = self.ollama_gemini_llm,
			# embedder={
			# 	"provider": "ollama",
			# 	"config": {
			# 		"model": "nomic-embed-text",
			# 		"base_url": "http://localhost:11434"
			# 	}
			# },
			# llm = self.gemini_llm,
			embedder={
        		"provider": "google",
        		"config": {
            		"model": "models/text-embedding-004",
            		"api_key": GEMINI_API_KEY,
        		}
    		},
		)

	@agent
	def market_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['market_agent'],
			tools=[SerperDevTool()],
			verbose=True,
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True,
		)

	@agent
	def translating_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['translating_agent'],
			verbose=True,
			llm = self.gemini_llm
		)
	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def crop_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['crop_agent_task'],
		)
	


	@task
	def climate_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['climate_agent_task'],
		)
	
	@task
	def soil_health_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['soil_health_agent_task'],
		)

	@task
	def fertilizer_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['fertilizer_agent_task'],
		)

	@task
	def scheme_policy_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['scheme_policy_agent_task'],
		)
	
	@task
	def market_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['market_agent_task'],

		)
	
	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)
	
	@task
	def translating_agent_task(self) -> Task:
		return Task(
			config=self.tasks_config['translating_agent_task'],
			llm = self.ollama_llm,
			output_file='tamil-report.md'
		)
	
	@crew
	def crew(self) -> Crew:
		"""Creates the MiniProjectTest crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# max_rpm=2 
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

# --------------

