from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from pypdf import PdfReader

import gradio as gr
import jinja2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
    trim_blocks=True,
    lstrip_blocks=True
)

SYSTEM_PROMPT = """You are an expert legal document assistant who helps generate professional legal documents.
Analyze the user's requirements and enhance them with proper legal language and structure."""

CharacteristicClass = Literal[
    "1a1",
    "2a1",
    "3a1",
]

@dataclass(frozen=True)
class CharacteristicDefinition:
    name: CharacteristicClass
    template_filename: str
    info: str


CHARACTERISTIC_DEFINITIONS: dict[CharacteristicClass, CharacteristicDefinition] = {
    "1a1": CharacteristicDefinition(
        name="1a1",
        template_filename="1a1_prompt.jinja",
        info="Classification type 1a1 for legal documents"
    ),
    "2a1": CharacteristicDefinition(
        name="2a1",
        template_filename="2a1_prompt.jinja",
        info="Classification type 2a1 for legal documents"
    ),
    "3a1": CharacteristicDefinition(
        name="3a1",
        template_filename="3a1_prompt.jinja",
        info="Classification type 3a1 for legal documents"
    ),
}

PROMPT_TEMPLATES = {
    characteristic: jinja_env.get_template(config.template_filename)
    for characteristic, config in CHARACTERISTIC_DEFINITIONS.items()
}

DEFAULT_CHARACTERISTIC: CharacteristicClass = "1a1"
CHARACTERISTIC_CHOICES: tuple[CharacteristicClass, ...] = tuple(CHARACTERISTIC_DEFINITIONS.keys())

CHARACTERISTIC_INFORMATION_BLOCK = "\n".join(
    f"- {characteristic}: {config.info}" 
    for characteristic, config in CHARACTERISTIC_DEFINITIONS.items()
)





def process_prompt(
    document_file,
    characteristic: CharacteristicClass
) -> str:
    """Process and generate a classification prompt using the selected template."""
    # Read PDF if provided
    document_text = ""
    if document_file is not None:
        reader = PdfReader(document_file.name)
        for page in reader.pages:
            document_text += page.extract_text()
    
    try:
        template = PROMPT_TEMPLATES[characteristic]
    except KeyError as error:
        raise ValueError(f"Unsupported characteristic: {characteristic}") from error

    # Render the template
    user_content = template.render(document_text=document_text)

    # Get AI-enhanced classification
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            }
        ],
        max_tokens=2000
    )
    
    return response.choices[0].message.content


def generate_prompt_handler(
    document_file,
    current_characteristic: str | None,
):
    """Main handler for prompt generation."""
    
    if document_file is None:
        return "Error: Please upload a document."
    
    prompt_text = process_prompt(
        document_file=document_file,
        characteristic=current_characteristic,
    )
    
    display_text = f"Selected characteristic: {current_characteristic}\n\n{prompt_text}"
    return display_text


# Gradio Interface
with gr.Blocks(title="Legal Document Classifier") as demo:
    gr.Markdown("# Legal Document Classifier")
    gr.Markdown("Classify legal documents using AI and Jinja templates.")
    
    with gr.Row():
        with gr.Column():
            document_input = gr.File(
                label="Upload Legal Document (PDF)",
                file_types=[".pdf"]
            )
            
            characteristic_dropdown = gr.Dropdown(
                choices=list(CHARACTERISTIC_CHOICES),
                value=DEFAULT_CHARACTERISTIC,
                label="Classification Characteristic",
                info="Choose the classification approach",
                interactive=True,
            )
            
            generate_button = gr.Button("Generate Classification", variant="primary")
            
        with gr.Column():
            prompt_output = gr.Textbox(
                label="Classification Result",
                lines=30,
                show_copy_button=True,
            )
    
    # Event handlers
    generate_button.click(
        generate_prompt_handler,
        inputs=[
            document_input,
            characteristic_dropdown,
        ],
        outputs=[prompt_output],
    )

if __name__ == "__main__":
    demo.launch()