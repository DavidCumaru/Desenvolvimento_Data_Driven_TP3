from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import re
load_dotenv('.env')

llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.environ['GEMINI_KEY'])

def is_valid_month(date_str):
    date_obj = datetime.strptime(date_str, "%d/%m/%Y")
    return date_obj.month in [8, 10]

def consultar_agenda(especialidade, data):
    if not is_valid_month(data):
        return f"Desculpe, só há disponibilidade para os meses de agosto e outubro. A data fornecida ({data}) não é válida."
    return f"Há disponibilidade para a especialidade {especialidade} em {data}."

def agendar_consulta(medico, data, paciente):
    if not is_valid_month(data):
        return f"Desculpe, só é possível agendar para os meses de agosto e outubro. A data fornecida ({data}) não é válida."
    return f"A consulta com o médico {medico} foi agendada para {data}."

agenda_tool = Tool(
    name="Consulta de Agenda",
    func=consultar_agenda,
    description="Consulta a disponibilidade de médicos e especialidades médicas"
)

agendamento_tool = Tool(
    name="Agendamento de Consulta",
    func=agendar_consulta,
    description="Agendamento de consulta médica com base na disponibilidade"
)

tools = [agenda_tool, agendamento_tool]
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_with_memory = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True)

st.title("Agendamento Médico Inteligente")

pergunta = st.text_input("Pergunte sobre o agendamento ou a disponibilidade:")

if pergunta:
    match = re.search(r'(\d{1,2}) de (agosto|outubro)', pergunta.lower())
    if match:
        dia = match.group(1)
        mes = match.group(2)
        if mes == 'agosto':
            data = f"{dia}/08/2024" 
        elif mes == 'outubro':
            data = f"{dia}/10/2024"
        elif mes != 'agosto' or mes != 'outubro':
            print('Só temos disponibilidade para o mês de agosto ou outubro')
        especialidade = "geral"

        resposta = consultar_agenda(especialidade, data)
    else:
        resposta = "Desculpe, não consegui entender a data solicitada. Por favor, use o formato 'dia X de agosto' ou 'dia X de outubro'."
    st.write(resposta)
