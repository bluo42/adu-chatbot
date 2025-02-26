import os
import streamlit as st
import openai
from openai import OpenAI

import os

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# --- Configuration ---
API_KEY = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=API_KEY)

# Directories for PDF files.
LETTERS_DIR = os.path.join("data", "Letters")
ORDINANCES_DIR = os.path.join("data", "Ordinances")
VECTOR_STORE_ID_FILE = "vector_store_id.txt"
VS_NAME = "ADU Permit Vector Store"

# -------------------------------
# Sidebar: Role Selection
# -------------------------------
role_selection = st.sidebar.radio(
    "Select Response Role",
    options=["Applicant", "Planner"],
    index=0,
    help="Choose whether to generate a persuasive permit applicant response or an antagonistic planner response."
)

# -------------------------------
# Helper: Build Assistant Instruction Based on Role
# -------------------------------
def get_instructions(role: str) -> str:
    if role == "Applicant":
        return (
            "You are a chatbot for ADU permit discussions. Your role is that of an applicant "
            "advocating for an ADU permit by persuasively citing the city's ADU ordinance. "
            "For example, if asked, 'I'm preparing my application for an ADU permit. How does the ordinance justify an increased number of ADUs in my neighborhood?', "
            "you should respond with a persuasive argument such as: 'According to section 3.2 of the ADUHandbookUpdate.pdf, the ordinance allows...'. "
            "The uploaded documents include letters and ordinances, with the ADUHandbookUpdate.pdf (the statewide file) taking precedence over individual files. "
            "Whenever possible, provide specific citations to ordinance sections that support the permit application."
        )
    else:  # Planner
        return (
            "You are a chatbot for ADU permit discussions. Your role is that of a city planner who is skeptical about an increased number of ADUs. "
            "Your responses should be antagonistic, highlighting potential negative impacts of ADUs and questioning the applicant's interpretation of the ordinance. "
            "For example, if asked, 'I'm preparing my application for an ADU permit. How does the ordinance justify an increased number of ADUs in my neighborhood?', "
            "you should respond with a critical counterargument such as: 'The ordinance in section 3.2 of the ADUHandbookUpdate.pdf is ambiguous and does not fully address community impacts.' "
            "The uploaded documents include letters and ordinances, with the ADUHandbookUpdate.pdf (the statewide file) taking precedence over individual files. "
            "Whenever possible, provide specific citations to ordinance sections that support your perspective."
        )

# -------------------------------
# Initialize Session State Variables
# -------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Only create/upload once.
if "current_role" not in st.session_state:
    st.session_state.current_role = role_selection  # Track current role for comparison.
if "assistant" not in st.session_state:
    assistant = client.beta.assistants.create(
        name="ADU Permit Chatbot Assistant",
        instructions=get_instructions(role_selection),
        model="gpt-4o",
        tools=[{"type": "file_search"}],
    )
    st.session_state.assistant = assistant
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I assist you with your ADU permit application?"}
    ]

# -------------------------------
# Update Assistant Instructions if Role Changes
# -------------------------------
if st.session_state.current_role != role_selection:
    new_instructions = get_instructions(role_selection)
    assistant = client.beta.assistants.update(
        assistant_id=st.session_state.assistant.id,
        instructions=new_instructions
    )
    st.session_state.assistant = assistant
    st.session_state.current_role = role_selection

# -------------------------------
# Helper: Get PDF File Paths
# -------------------------------
def get_pdf_file_paths():
    """
    Crawls the LETTERS_DIR and ORDINANCES_DIR for PDF files.
    Prioritizes the ADUHandbookUpdate.pdf in the ordinances directory.
    """
    pdf_paths = []
    
    # Process Letters directory.
    if os.path.exists(LETTERS_DIR):
        for filename in os.listdir(LETTERS_DIR):
            if filename.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(LETTERS_DIR, filename))
    
    # Process Ordinances directory.
    if os.path.exists(ORDINANCES_DIR):
        statewide_file = "ADUHandbookUpdate.pdf"
        statewide_path = os.path.join(ORDINANCES_DIR, statewide_file)
        # Insert statewide file at the beginning if it exists.
        if os.path.exists(statewide_path):
            pdf_paths.insert(0, statewide_path)
        # Add any other PDFs in the ordinances directory.
        for filename in os.listdir(ORDINANCES_DIR):
            if filename.lower().endswith(".pdf") and filename != statewide_file:
                pdf_paths.append(os.path.join(ORDINANCES_DIR, filename))
    return pdf_paths

# -------------------------------
# Generate Response Function
# -------------------------------
def generate_response(messages):
    """
    Sends the conversation messages to the persistent assistant,
    creates a thread, polls for completion, and returns the assistant's reply.
    """
    assistant = st.session_state.assistant
    thread = client.beta.threads.create(messages=messages)
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please answer the user's query based on the conversation context.",
        max_prompt_tokens=20000,
        max_completion_tokens=5000
    )
    messages_list = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    if messages_list:
        return messages_list[0].content[0].text.value.strip()
    else:
        return "No response received from the assistant."

# -------------------------------
# Create/Upload Vector Store (Persisted Across Sessions)
# -------------------------------
if st.session_state.vector_store is None:
    vector_store = None
    # Try loading vector store ID from st.secrets.
    if "VECTOR_STORE_ID" in st.secrets:
        stored_vector_store_id = st.secrets["VECTOR_STORE_ID"]['vs_id']
        try:
            vector_store = client.beta.vector_stores.retrieve(stored_vector_store_id)
            st.sidebar.write("Loaded persisted vector store ID from secrets:", stored_vector_store_id)
        except Exception as e:
            st.sidebar.write("Failed to load persisted vector store from secrets. Creating a new one. Error:", e)
    # If not found or failed to load, create a new vector store.
    if vector_store is None:
        vector_store = client.beta.vector_stores.create(name=VS_NAME)
        pdf_paths = get_pdf_file_paths()
        if pdf_paths:
            file_streams = [open(path, "rb") for path in pdf_paths]
            client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=file_streams
            )
            st.sidebar.write("Uploaded files:", [os.path.basename(p) for p in pdf_paths])
        else:
            st.sidebar.write("No PDF files found in the specified directories.")
        st.sidebar.write("New vector store ID (please update your secrets.toml manually):", vector_store.id)
    st.session_state.vector_store = vector_store
    # Update the assistant's tool resources to use the vector store.
    assistant = client.beta.assistants.update(
        assistant_id=st.session_state.assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [st.session_state.vector_store.id]}}
    )
    st.session_state.assistant = assistant

# -------------------------------
# Main Chatbot Interface
# -------------------------------
st.title("💬 ADU Permit Chatbot")

# Display conversation history.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input and response generation.
if prompt := st.chat_input("Your message:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    try:
        bot_response = generate_response(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.chat_message("assistant").write(bot_response)
    except Exception as e:
        error_msg = f"Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.chat_message("assistant").write(error_msg)
