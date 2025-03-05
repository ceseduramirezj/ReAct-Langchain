from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool
from typing import Union, List
from langchain.agents.format_scratchpad import format_log_to_str

from callbacks import AgentCallbackHandler

import os

# Con esta anotación indicamos que una función es un 'tool' que puede ser utilizado por los agentes
# Los 'tools' son funciones que pueden ser utilizadas por los agentes para realizar tareas específicas
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    ) # striping away non alphabetic characters just in case
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
        
    raise ValueError(f"Tool with name {tool_name} not found in tools")

if __name__ == '__main__':
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    # Esta es la plantilla de nuestro prompt que utilizará nuestro agente.
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    # A nuestra plantilla le indicamos las tools a disposición del agente y los nombres de las tools
    # que se pueden utilizar en el campo
    prompt = PromptTemplate.from_template(template= template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))
    
    # Al crear la referencia al LLM podemos indicar en el parámetro 'stop' las cadenas de texto que van a detener la generación
    # de texto cuando el modelo las genere, ojo que en la respuesta no se va a incluir la cadena de texto especificada. 
    # En este caso, se detendrá la generación cuando el modelo genere la cadena "\nObservation" y no se incluye esta cadena en la respuesta.
    # De esta forma controlamos la generación y podemos ahorrar costes.

    # IMPORTANTE temperature=0 para que el modelo no genere palabras aleatorias
    # En el campo 'callbacks' indicamos el objeto correspondiente para manejar los eventos de la conversación
    # En este caso mostrar el prompt enviado y el texto generado por el modelo
    llm = AzureChatOpenAI(temperature=0, 
                          stop=["\nObservation"], 
                          azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
                          callbacks=[AgentCallbackHandler()])
    
    # Almacenará por cada repetición del Agente los pasos intermedios, la herramienta que elige el LLM a utilizar
    # y el resultado de la herramienta
    intermediate_steps = []

    # Encadenacion se alimenta el LLM con el prompt
    # Creamos nuestro agente con el template, el LLM y el parser que vamos a utilizar para interpretar la salida del agente
    # La clase parser ReActSingleInputOutputParser (funciona con un único tool) es una clase que se encarga de interpretar 
    # la salida del agente y determinar si el agente debe realizar una acción (AgentAction) o si debe devolver 
    # una respuesta final (AgentFinish)
    agent = {
            "input": lambda x: x["input"], 
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
        } | prompt | llm | ReActSingleInputOutputParser()

    agent_step = ""

    # Se lanza el Agente hasta que devuelva una respuesta final es decir un objeto de tipo AgentFinish
    # Para ello en cada iteración hasta que se cumpla la condición se invoca el agente con el método invoke
    # Y se evalua cada iteración con una variable de tipo AgentAction o AgentFinish
    while not isinstance(agent_step, AgentFinish):

        # Invocamos el Agente que puede devolver AgentAction o AgentFinish
        # Incluimos en cada repetición los pasos intermedios para proporcionar el resultado de haber elegido
        # y utilizado una herramienta, en caso sea un resultado lógico para el agente pues este llegará
        # a la respuesta final
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {"input": "What is the length in characters of the text DOG?'",
            "agent_scratchpad": intermediate_steps
            }     
        )
        print(agent_step)

        # Si el resultado es un objeto de tipo AgentAction se procede a ejecutar la acción
        # y se almacena tanto la elección de herramienta como el resultado en la lista de pasos intermedios
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
        
            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation)))
        
    # Cuando se llega a la respuesta final se imprime el resultado
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)