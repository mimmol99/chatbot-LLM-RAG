import gradio as gr

class GUI():


    def __init__(self,answer_generator):

        self.retriever = answer_generator.get_retriever()
        self.answer_generator = answer_generator
        self.start_chat()


    def get_name(self):
        def submit_name(name):
            return name

        name_input = gr.Textbox(label="Enter your name")
        name_interface = gr.Interface(fn=submit_name, inputs=name_input, outputs="text", live=False)

        result = name_interface.launch(share=False)
        return result


    def start_chat(self):

        def chat(inp):
            user_input = inp
            if not user_input:
                return None
            answer = self.answer_generator.answer_prompt(user_input)
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

        
    


