import asyncio
import streamlit as st
from interact.base import Cascade, Handler, Message
from interact.handlers import OpenAiLLM

class FunctionPrompt(Handler):
    role = "FirstAidRecommender"
    fprompt = """Generate first aid recommendations for a given query. Display the ingredients needed for the treatment of the query too. 
    If the given query states a severe issue that should be treated by a doctor only, suggest a specialized doctor who can treat it.
    \nQuery: {user_query}."""
    
    async def process(self, msg: Message, csd: Cascade) -> str:
        user_input = msg.primary
        fprompt = self.fprompt.format(user_query=user_input)    
        return fprompt

def main():
    st.title("First Aid Recommender")
    query = st.text_input("Enter your query:")
        
    if st.button("Show"):
        if query:
            # Create the handler cascade
            generator = FunctionPrompt() >> OpenAiLLM()
            # Run the cascade asynchronously
            result = asyncio.run(generator.start(Message(query)))
            # Get the response from the last message in the cascade
            answer = result.last_msg.primary
            # Display the response
            st.write(answer)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
