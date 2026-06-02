import streamlit as st

st.set_page_config(
    page_title="Simple AI App",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Simple AI App")
st.write("This is the first deployed version. AI integration will be added later.")

user_input = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_input.strip():
        st.success("App is working successfully!")
        st.write("You entered:")
        st.info(user_input)

        st.write("Dummy Response:")
        st.write("This is a sample response. Later, this will come from ChatGPT API.")
    else:
        st.warning("Please enter some text first.")