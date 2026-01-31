import gradio as gr
import os
import shutil
import torch
import run
import config
import base64
import mimetypes
import re

from pathlib import Path

"""Global variables to store model and state"""
model = None
tokenizer = None
current_output_dir = None


def initialize_model(int4: bool, device: str):
    """
    Initialize the model once and store in global variables.

    :params int4: Whether to use 4-bit quantization
    :params device: Device to load model on

    Returns:
        str: Status message
    """
    global model, tokenizer
    try:
        model, tokenizer = run.load_model(int4=int4, device=device)
        return f"✓ Model loaded successfully on {device}"
    except Exception as e:
        return f"✗ Error loading model: {str(e)}"


def process_input(
    file_input,
    num_pages: int,
    prompt: str,
    base_size: int,
    image_size: int,
    max_pages: int,
    progress=gr.Progress()
):
    """
    Process uploaded PDF or image file through OCR pipeline.

    :params file_input: Uploaded file path (PDF or image)
    :params num_pages: Number of pages to process for PDFs
    :params prompt: Prompt type ('grounding' or 'free ocr')
    :params base_size: Base size for inference
    :params image_size: Image size for inference
    :params max_pages: Maximum pages to process
    :params progress: Gradio progress tracker

    Returns:
        tuple: (status_message, merged_markdown_content, merged_markdown_path)
    """
    global model, tokenizer, current_output_dir

    if model is None or tokenizer is None:
        return 'Error: Model not initialized. Please load the model first.', '', None

    if file_input is None:
        return 'Error: No file uploaded.', '', None

    try:
        file_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
        file_ext = Path(file_path).suffix.lower()

        progress(0, desc='Preparing files...')

        """Determine image files based on input type"""
        if file_ext in ['.pdf']:
            progress(0.1, desc='Converting PDF to images...')
            image_files = run.read_pdf(file_path)
            if num_pages and num_pages > 0:
                image_files = image_files[:num_pages]
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            image_files = [file_path]
        else:
            return f"Error: Unsupported file format {file_ext}", '', None

        """Run inference with progress tracking"""
        num_images = len(image_files)
        progress(0.2, desc=f"Starting inference on {num_images} image(s)...")

        output_dir = run.run_inference(
            image_files,
            file_path=file_path,
            model=model,
            tokenizer=tokenizer,
            output_dir=None,
            prompt=prompt,
            base_size=base_size,
            image_size=image_size,
            crop_mode=True,
            save_results=True,
            max_pages=max_pages
        )

        progress(0.8, desc='Merging markdown files...')

        """Merge markdowns"""
        merged_file = run.merge_markdowns(output_dir, merged_markdown=config.MERGED_OUTPUT_NAME)
        current_output_dir = output_dir

        """Read merged markdown and convert image paths to base64 data URIs"""
        with open(merged_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        def convert_images_to_base64(md):
            """
            Convert all image paths in markdown to base64 data URIs.

            :params md: Markdown content with image paths

            Returns:
                str: Markdown with images as data URIs
            """
            def repl(match):
                img_path = match.group(1)
                # Handle both absolute and relative paths
                if os.path.isabs(img_path):
                    img_file = img_path
                else:
                    img_file = os.path.join(output_dir, img_path)
                
                if not os.path.exists(img_file):
                    return match.group(0)
                
                mime, _ = mimetypes.guess_type(img_file)
                mime = mime or 'application/octet-stream'
                try:
                    with open(img_file, 'rb') as imf:
                        b64 = base64.b64encode(imf.read()).decode('ascii')
                    return f'![](data:{mime};base64,{b64})'
                except Exception:
                    return match.group(0)

            # Match patterns like ![](path/to/image.jpg) or ![alt](path)
            return re.sub(r'!\[[^\]]*\]\(([^)]+)\)', repl, md)

        markdown_content = convert_images_to_base64(markdown_content)

        progress(1.0, desc='Processing complete!')

        status = f"✓ Successfully processed {num_images} image(s) from {Path(file_path).name}"
        return status, markdown_content, merged_file

    except Exception as e:
        return f"✗ Error during processing: {str(e)}", '', None

def clear_outputs():
    """
    Clear all output files and reset state.

    Returns:
        tuple: (status_message, empty_markdown, None)
    """
    global current_output_dir
    try:
        if os.path.exists('outputs'):
            shutil.rmtree('outputs')
            os.makedirs('outputs', exist_ok=True)
        current_output_dir = None
        return '✓ Outputs cleared', '', None
    except Exception as e:
        return f"✗ Error clearing outputs: {str(e)}", '', None


def get_device_options():
    """
    Get available device options based on system.

    Returns:
        list: Available device options
    """
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.insert(0, 'cuda')
    return devices


def create_interface():
    """
    Create and configure the Gradio interface.

    Returns:
        gr.Blocks: Configured Gradio application
    """
    with gr.Blocks(title='DeepSeek OCR 2 Inference', theme=gr.themes.Soft()) as app:
        gr.Markdown('# DeepSeek OCR 2 Inference')
        gr.Markdown('Convert PDFs and images to markdown using DeepSeek OCR 2 model')

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('### Model Configuration')

                int4_checkbox = gr.Checkbox(
                    label='Use 4-bit Quantization',
                    value=config.DEFAULT_INT4,
                    info='Reduces memory usage but may impact quality'
                )

                device_radio = gr.Radio(
                    choices=get_device_options(),
                    value='cuda' if 'cuda' in get_device_options() else 'cpu',
                    label='Device',
                    info='Select processing device'
                )

                load_btn = gr.Button('Load Model', variant='primary', size='lg')
                model_status = gr.Textbox(
                    label='Model Status',
                    interactive=False,
                    value='Model not loaded'
                )

                load_btn.click(
                    initialize_model,
                    inputs=[int4_checkbox, device_radio],
                    outputs=model_status
                )

            with gr.Column(scale=2):
                gr.Markdown('### Input & Processing Settings')

                file_input = gr.File(
                    label='Upload PDF or Image',
                    type='filepath'
                )

                with gr.Row():
                    num_pages_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1,
                        label='Max Pages to Process (PDFs)',
                        info='Limits processing for large PDFs'
                    )

                    prompt_radio = gr.Radio(
                        choices=['grounding', 'free ocr'],
                        value='grounding',
                        label='Prompt Type',
                        info='grounding: Structured document OCR, free ocr: Basic OCR'
                    )

                with gr.Row():
                    base_size_slider = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=config.BASE_SIZE,
                        step=256,
                        label='Base Size',
                        info='Model inference base size'
                    )

                    image_size_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=config.IMAGE_SIZE,
                        step=128,
                        label='Image Size',
                        info='Model inference image size'
                    )

        with gr.Row():
            process_btn = gr.Button('Process', variant='primary', size='lg')
            clear_btn = gr.Button('Clear Outputs', variant='stop', size='lg')

        process_status = gr.Textbox(
            label='Processing Status',
            interactive=False
        )

        gr.Markdown('### Results')

        with gr.Tabs():
            with gr.Tab('Rendered Output'):
                markdown_output = gr.Markdown(
                    label='Merged Markdown',
                    value='*Output will appear here*'
                )

            with gr.Tab('Raw Markdown'):
                raw_markdown = gr.Textbox(
                    label='Raw Markdown Content',
                    lines=20,
                    interactive=False
                )

            with gr.Tab('Download'):
                download_file = gr.File(
                    label='Download Merged Markdown',
                    type='filepath',
                    interactive=False
                )

        """Event handlers"""
        process_btn.click(
            process_input,
            inputs=[
                file_input,
                num_pages_slider,
                prompt_radio,
                base_size_slider,
                image_size_slider,
                num_pages_slider
            ],
            outputs=[process_status, markdown_output, download_file],
            api_name='process'
        ).then(
            lambda x: x,
            inputs=[markdown_output],
            outputs=[raw_markdown]
        )

        clear_btn.click(
            clear_outputs,
            outputs=[process_status, markdown_output, download_file]
        )

        with gr.Accordion('Tips', open=False):
            gr.Markdown("""
            - Load the model once, then process multiple files
            - Use 4-bit quantization on low-memory systems (GPU with <24GB)
            - Increase base_size for better quality (slower), decrease for speed
            - Use 'grounding' prompt for structured documents, 'free ocr' for general text
            """)

    return app


if __name__ == '__main__':
    app = create_interface()
    app.launch(
        share=True, 
        server_name='127.0.0.1', 
        server_port=7860    
    )
