import gradio as gr

class GUI():


    def __init__(self,answer_generator):

        self.answer_generator = answer_generator
        self.start_chat()

    def start_chat(self):

        def chat(inp):
            user_input = inp
            if not user_input:
                return None
            answer = self.answer_generator.answer_prompt(user_input)
            #self.answer_generator.get_store()
            return f"\n{answer}"

        interface = gr.Interface(
            fn=chat,
            inputs=gr.Textbox(placeholder="Type your question here..."),
            outputs=gr.Textbox(value=""),
            title="Chat with me",
            description="Ask me anything!",
            elem_id="chat-container",
        )

        # Display the interface
        interface.launch()

        
    


