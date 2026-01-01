import streamlit as st
from pathlib import Path
from graph import app
from rag import ingest_pdf_to_qdrant
import hashlib

# Page configuration
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()
if "processing" not in st.session_state:
    st.session_state.processing = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .chat-container {
        height: calc(100vh - 280px);
        overflow-y: auto;
        padding: 1rem 0;
    }
    .main-content {
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow: hidden;
    }
    .chat-section {
        flex: 1;
        overflow-y: auto;
        padding-bottom: 20px;
    }
    .input-section {
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        flex-shrink: 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-header">🤖 Agentic AI Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "Ask me anything! I can help with weather information, answer questions, and more."
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(
        "This app uses an AI workflow to process your queries intelligently."
    )

    # File upload section
    st.divider()
    st.header("📄 Document Upload")
    st.markdown("Upload PDF documents to enrich the knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", label_visibility="collapsed", key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        # Create a unique hash of the file to check if it's already been uploaded
        file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
        
        if file_hash not in st.session_state.uploaded_files:
            # Create documents directory if it doesn't exist
            documents_dir = Path("./documents")
            documents_dir.mkdir(exist_ok=True)

            # Save the uploaded file
            file_path = documents_dir / uploaded_file.name

            with st.spinner(f"Uploading {uploaded_file.name}..."):
                try:
                    # Save file to documents directory
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success(f"✅ File saved: {uploaded_file.name}")

                    # Start ingestion
                    with st.spinner(
                        f"🔄 Ingesting {uploaded_file.name} into Qdrant..."
                    ):
                        ingest_pdf_to_qdrant(
                            str(file_path),
                            collection_name="documents",
                            chunk_size=1000,
                            chunk_overlap=200,
                        )

                    # Mark file as uploaded
                    st.session_state.uploaded_files.add(file_hash)
                    
                    st.success(
                        f"✅ Successfully ingested {uploaded_file.name} to the knowledge base!"
                    )
                    st.balloons()
                    
                    # Reset file uploader by incrementing key
                    st.session_state.uploader_key += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.error(f"Details: {type(e).__name__}")
        else:
            st.info(f"ℹ️ {uploaded_file.name} has already been uploaded.")

# Main content
st.divider()

# Chat section
with st.container():
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown("👤 **You**")
                st.markdown(f"{message['content']}")
            else:
                st.markdown("🤖 **Assistant**")
                st.markdown(f"{message['content']}")
            
            # Add separator between messages (except after the last message)
            if i < len(st.session_state.chat_history) - 1:
                st.divider()
    else:
        st.markdown('<p style="color: gray; text-align: center;">Start a conversation...</p>', unsafe_allow_html=True)

st.divider()

# Input form with disabled state during processing
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'What's the weather like in New York today?' or 'Tell me about machine learning'",
        label_visibility="collapsed",
        disabled=st.session_state.processing,
    )

with col2:
    submit_button = st.button(
        "🚀 Send",
        use_container_width=True,
        disabled=st.session_state.processing,
    )

# Process query only when submit button is clicked
if submit_button and user_query:
    # Set processing flag to disable button
    st.session_state.processing = True
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    
    with st.spinner("Processing your query..."):
        try:
            # Invoke the workflow
            result = app.invoke({"user_query": user_query})

            # Extract the final answer
            final_answer = result.get("final_answer") or result.get("llm_response", "No response generated.")
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_answer
            })

            # Debug info (collapsible)
            with st.expander("📊 Debug Information"):
                if result.get("retrieved_chunks"):
                    st.subheader("Retrieved Chunks")
                    for i, chunk in enumerate(result["retrieved_chunks"], 1):
                        st.text(f"{i}. {chunk}")

                if result.get("errors"):
                    st.subheader("⚠️ Errors")
                    for error in result["errors"]:
                        st.error(error)

                if result.get("trace_id"):
                    st.subheader("Trace ID")
                    st.code(result["trace_id"])

                st.subheader("Full State")
                st.json({k: v for k, v in result.items() if v is not None})

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error(f"Details: {type(e).__name__}")
        
        finally:
            # Reset processing flag after query is done
            st.session_state.processing = False
            # Rerun to display updated chat history immediately
            st.rerun()

elif submit_button:
    st.warning("Please enter a query.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; margin-top: 2rem;'>
    Built with Streamlit and LangGraph | Powered by AI 🚀
    </div>
    """,
    unsafe_allow_html=True,
)
