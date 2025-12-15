import streamlit as st
import requests
import json

st.set_page_config(page_title="Nokcha - Fraud Q&A Chatbot", page_icon="🤖")

def main():
    st.title("Nokcha - Fraud Q&A Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Send chat history to the streaming endpoint
                response = requests.post(
                    "http://engine:8000/stream",
                    json=st.session_state.messages,
                    stream=True
                )
                response.raise_for_status()  # Raise an exception for bad status codes

                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8')
                        try:
                            # 3. Parse the line as a JSON object.
                            # The FastAPI endpoint yields {"content": "...", ...}
                            chunk_data = json.loads(decoded_chunk)
                            # 4. Extract the streamed content.
                            content = chunk_data.get('content', '')

                            # Handle a completion signal, if your FastAPI uses one (e.g., {"done": true})
                            if chunk_data.get("done"):
                                break
                            
                            full_response += content
                            message_placeholder.markdown(full_response + "▌") # Add a cursor for effect
                            
                        except json.JSONDecodeError:
                            # Handle case where chunk is incomplete or malformed JSON
                            continue
                message_placeholder.markdown(full_response)
            
            except requests.exceptions.RequestException as e:
                full_response = f"Error: {e}"
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()

