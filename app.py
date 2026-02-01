import gradio as gr
import os
import shutil
import torch
import run
import config
import base64
import mimetypes
import re
import json
import markdown
import logging

from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

"""Global variables to store model and state"""
model = None
tokenizer = None
current_output_dir = None
session_history = []


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
        return f'✓ Model loaded successfully on {device}'
    except Exception as e:
        return f'✗ Error loading model: {str(e)}'


def convert_images_to_base64(md, output_dir):
    """
    Convert all image paths in markdown to base64 data URIs.

    :params md: Markdown content with image paths
    :params output_dir: Output directory for resolving relative paths

    Returns:
        str: Markdown with images as data URIs
    """
    def repl(match):
        img_path = match.group(1)
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

    return re.sub(r'!\[[^\]]*\]\(([^)]+)\)', repl, md)


def format_stats(stats):
    """
    Format statistics dict for display.

    :params stats: Stats dictionary from run_inference

    Returns:
        str: Formatted stats string
    """
    if not stats:
        return ''

    lines = [
        f"📊 **Processing Statistics**",
        f"- Total Pages: {stats.get('total_pages', 'N/A')}",
        f"- Total Time: {stats.get('total_time', 0):.2f}s",
        f"- Avg Time/Page: {stats.get('avg_time_per_page', 0):.2f}s",
    ]

    if stats.get('page_times'):
        lines.append(f"- Fastest Page: {min(stats['page_times']):.2f}s")
        lines.append(f"- Slowest Page: {max(stats['page_times']):.2f}s")

    return '\n'.join(lines)


def export_to_format(markdown_content, export_format, output_dir):
    """
    Export markdown content to different formats.

    :params markdown_content: Raw markdown content
    :params export_format: Target format ('markdown', 'html', 'json')
    :params output_dir: Output directory for saving exported file

    Returns:
        str: Path to exported file
    """
    if not markdown_content or not output_dir:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if export_format == 'html':
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <title>OCR Output</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        img {{ max-width: 100%; height: auto; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
        code {{ background: #f4f4f4; padding: 2px 5px; }}
    </style>
</head>
<body>
{markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])}
</body>
</html>"""
        export_path = os.path.join(output_dir, f'export_{timestamp}.html')
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return export_path

    elif export_format == 'json':
        json_content = {
            'timestamp': timestamp,
            'content': markdown_content,
            'lines': markdown_content.split('\n'),
            'word_count': len(markdown_content.split()),
            'char_count': len(markdown_content)
        }
        export_path = os.path.join(output_dir, f'export_{timestamp}.json')
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, indent=2, ensure_ascii=False)
        return export_path

    else:  # markdown
        export_path = os.path.join(output_dir, f'export_{timestamp}.md')
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        return export_path


def add_to_history(file_name, output_dir, stats):
    """Add processed file to session history."""
    global session_history
    session_history.append({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'file': file_name,
        'output_dir': output_dir,
        'pages': stats.get('total_pages', 0),
        'time': f"{stats.get('total_time', 0):.1f}s"
    })


def get_history_display():
    """Generate history display markdown."""
    global session_history
    if not session_history:
        return '*No files processed yet*'

    rows = ['| Time | File | Pages | Duration |', '|------|------|-------|----------|']
    for item in reversed(session_history[-10:]):
        rows.append(f"| {item['timestamp']} | {item['file']} | {item['pages']} | {item['time']} |")

    return '\n'.join(rows)


def process_input(
    file_input,
    num_pages: int,
    prompt: str,
    base_size: int,
    image_size: int,
    export_format: str,
    progress=gr.Progress()
):
    """
    Process uploaded PDF or image file through OCR pipeline.

    :params file_input: Uploaded file path (PDF or image)
    :params num_pages: Number of pages to process for PDFs
    :params prompt: Prompt type ('grounding' or 'free ocr')
    :params base_size: Base size for inference
    :params image_size: Image size for inference
    :params export_format: Export format for results
    :params progress: Gradio progress tracker

    Returns:
        tuple: (status, markdown_content, raw_content, stats, download)
    """
    global model, tokenizer, current_output_dir

    if model is None or tokenizer is None:
        return ('Error: Model not initialized. Please load the model first.', 
               '', '', '', None)

    if file_input is None:
        return ('Error: No file uploaded.', '', '', '', None)

    try:
        file_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name

        progress(0, desc='Preparing files...')

        if file_ext in ['.pdf']:
            progress(0.1, desc='Converting PDF to images...')
            image_files = run.read_pdf(file_path)
            if num_pages and num_pages > 0:
                image_files = image_files[:num_pages]
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            image_files = [file_path]
        else:
            return (f'Error: Unsupported file format {file_ext}', '', '', '', None)

        num_images = len(image_files)
        progress(0.2, desc=f'Starting inference on {num_images} image(s)...')

        output_dir, stats = run.run_inference(
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
            max_pages=num_pages
        )

        progress(0.8, desc='Merging markdown files...')

        merged_file = run.merge_markdowns(output_dir, merged_markdown=config.MERGED_OUTPUT_NAME)
        current_output_dir = output_dir

        with open(merged_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        markdown_content = convert_images_to_base64(markdown_content, output_dir)

        export_path = export_to_format(markdown_content, export_format, output_dir)

        add_to_history(file_name, output_dir, stats)

        stats_display = format_stats(stats)

        progress(1.0, desc='Processing complete!')

        status = f"✓ Successfully processed {num_images} image(s) from {file_name}"
        return (status, markdown_content, markdown_content, stats_display, export_path)

    except Exception as e:
        logger.error(f'Error during processing: {e}')
        return (f'✗ Error during processing: {str(e)}', '', '', '', None)


def process_batch(
    folder_path: str,
    prompt: str,
    base_size: int,
    image_size: int,
    max_pages_per_file: int,
    progress=gr.Progress()
):
    """
    Process multiple files from a folder.

    :params folder_path: Path to folder containing PDFs/images
    :params prompt: Prompt type
    :params base_size: Base size for inference
    :params image_size: Image size for inference
    :params max_pages_per_file: Max pages per file
    :params progress: Gradio progress tracker

    Returns:
        tuple: (status, results_markdown, batch_stats)
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        return ('Error: Model not initialized.', '', '')

    if not folder_path or not os.path.isdir(folder_path):
        return ('Error: Invalid folder path.', '', '')

    supported_exts = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
    files = []
    for f in os.listdir(folder_path):
        if Path(f).suffix.lower() in supported_exts:
            files.append(os.path.join(folder_path, f))

    if not files:
        return ('No supported files found in folder.', '', '')

    total_files = len(files)
    all_results = []
    batch_stats = {
        'total_files': total_files,
        'processed': 0,
        'failed': 0,
        'total_pages': 0,
        'total_time': 0
    }

    for idx, file_path in enumerate(files):
        file_name = Path(file_path).name
        progress((idx / total_files), desc=f'Processing {file_name} ({idx+1}/{total_files})...')

        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.pdf':
                image_files = run.read_pdf(file_path)
                if max_pages_per_file:
                    image_files = image_files[:max_pages_per_file]
            else:
                image_files = [file_path]

            output_dir, stats = run.run_inference(
                image_files,
                file_path=file_path,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                base_size=base_size,
                image_size=image_size,
                max_pages=max_pages_per_file
            )

            run.merge_markdowns(output_dir)

            batch_stats['processed'] += 1
            batch_stats['total_pages'] += stats.get('total_pages', 0)
            batch_stats['total_time'] += stats.get('total_time', 0)

            all_results.append(f"✓ {file_name}: {stats.get('total_pages', 0)} pages in {stats.get('total_time', 0):.1f}s")

            add_to_history(file_name, output_dir, stats)

        except Exception as e:
            batch_stats['failed'] += 1
            all_results.append(f"✗ {file_name}: {str(e)}")

    progress(1.0, desc='Batch processing complete!')

    results_md = '\n'.join(all_results)
    stats_md = f"""📊 **Batch Statistics**
- Total Files: {batch_stats['total_files']}
- Processed: {batch_stats['processed']}
- Failed: {batch_stats['failed']}
- Total Pages: {batch_stats['total_pages']}
- Total Time: {batch_stats['total_time']:.1f}s"""

    return (f"✓ Batch complete: {batch_stats['processed']}/{total_files} files processed",
           results_md, stats_md)


def clear_outputs():
    """Clear all output files and reset state."""
    global current_output_dir, session_history
    try:
        if os.path.exists('outputs'):
            shutil.rmtree('outputs')
            os.makedirs('outputs', exist_ok=True)
        current_output_dir = None
        session_history = []
        return ('✓ Outputs cleared', '', '', '', None)
    except Exception as e:
        return (f'✗ Error clearing outputs: {str(e)}', '', '', '', None)


def get_device_options():
    """Get available device options based on system."""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.insert(0, 'cuda')
    return devices


def create_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(title='DeepSeek OCR 2', theme=gr.themes.Soft()) as app:
        gr.Markdown('# 🔍 DeepSeek OCR 2 Inference')
        gr.Markdown('Convert PDFs and images to markdown using DeepSeek OCR 2 model')

        with gr.Tabs() as main_tabs:
            # ==================== SINGLE FILE TAB ====================
            with gr.Tab('📄 Single File', id='single'):
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
                        gr.Markdown('### Input & Settings')

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
                                label='Max Pages (PDFs)',
                                info='Limits processing for large PDFs'
                            )

                            prompt_radio = gr.Radio(
                                choices=['grounding', 'free ocr'],
                                value='grounding',
                                label='Prompt Type'
                            )

                        with gr.Row():
                            base_size_slider = gr.Slider(
                                minimum=512,
                                maximum=2048,
                                value=config.BASE_SIZE,
                                step=256,
                                label='Base Size'
                            )

                            image_size_slider = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                value=config.IMAGE_SIZE,
                                step=128,
                                label='Image Size'
                            )

                        export_format = gr.Dropdown(
                            choices=['markdown', 'html', 'json'],
                            value='markdown',
                            label='Export Format'
                        )

                with gr.Row():
                    process_btn = gr.Button('🚀 Process', variant='primary', size='lg')
                    clear_btn = gr.Button('🗑️ Clear Outputs', variant='stop', size='lg')

                process_status = gr.Textbox(
                    label='Processing Status',
                    interactive=False
                )

                gr.Markdown('### Results')

                with gr.Tabs():
                    with gr.Tab('📝 Rendered Output'):
                        markdown_output = gr.Markdown(
                            label='Merged Markdown',
                            value='*Output will appear here*'
                        )

                    with gr.Tab('📋 Raw Markdown'):
                        with gr.Row():
                            raw_markdown = gr.Textbox(
                                label='Raw Markdown Content',
                                lines=15,
                                interactive=False
                            )
                        with gr.Row():
                            copy_btn = gr.Button('📋 Copy to Clipboard', size='sm')

                    with gr.Tab('📊 Statistics'):
                        stats_display = gr.Markdown(
                            value='*Statistics will appear after processing*'
                        )

                    with gr.Tab('💾 Download'):
                        download_file = gr.File(
                            label='Download Exported File',
                            type='filepath',
                            interactive=False
                        )

                # Copy button JS action
                copy_btn.click(
                    None,
                    inputs=[raw_markdown],
                    outputs=[],
                    js="(text) => { navigator.clipboard.writeText(text); }"
                )

                # Event handlers
                process_btn.click(
                    process_input,
                    inputs=[
                        file_input,
                        num_pages_slider,
                        prompt_radio,
                        base_size_slider,
                        image_size_slider,
                        export_format
                    ],
                    outputs=[
                        process_status,
                        markdown_output,
                        raw_markdown,
                        stats_display,
                        download_file
                    ],
                    api_name='process'
                )

                clear_btn.click(
                    clear_outputs,
                    outputs=[
                        process_status,
                        markdown_output,
                        raw_markdown,
                        stats_display,
                        download_file
                    ]
                )

            # ==================== BATCH PROCESSING TAB ====================
            with gr.Tab('📁 Batch Processing', id='batch'):
                gr.Markdown('### Process Multiple Files from a Folder')

                folder_input = gr.Textbox(
                    label='Folder Path',
                    placeholder='/path/to/folder/with/pdfs',
                    info='Enter absolute path to folder containing PDFs and images'
                )

                with gr.Row():
                    batch_prompt = gr.Radio(
                        choices=['grounding', 'free ocr'],
                        value='grounding',
                        label='Prompt Type'
                    )

                    batch_max_pages = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label='Max Pages per File'
                    )

                with gr.Row():
                    batch_base_size = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=config.BASE_SIZE,
                        step=256,
                        label='Base Size'
                    )

                    batch_image_size = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=config.IMAGE_SIZE,
                        step=128,
                        label='Image Size'
                    )

                batch_btn = gr.Button('🚀 Process Batch', variant='primary', size='lg')

                batch_status = gr.Textbox(
                    label='Batch Status',
                    interactive=False
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown('### Results')
                        batch_results = gr.Markdown(
                            value='*Results will appear here*'
                        )

                    with gr.Column():
                        gr.Markdown('### Batch Statistics')
                        batch_stats_display = gr.Markdown(
                            value='*Statistics will appear here*'
                        )

                batch_btn.click(
                    process_batch,
                    inputs=[
                        folder_input,
                        batch_prompt,
                        batch_base_size,
                        batch_image_size,
                        batch_max_pages
                    ],
                    outputs=[batch_status, batch_results, batch_stats_display]
                )

            # ==================== SESSION HISTORY TAB ====================
            with gr.Tab('📜 History', id='history'):
                gr.Markdown('### Session History')
                gr.Markdown('*Files processed in this session*')

                history_display = gr.Markdown(
                    value=get_history_display()
                )

                refresh_history_btn = gr.Button('🔄 Refresh', size='sm')
                refresh_history_btn.click(
                    get_history_display,
                    outputs=[history_display]
                )

        with gr.Accordion('💡 Tips', open=False):
            gr.Markdown("""
            - **Load once, process many**: Load the model once, then process multiple files
            - **4-bit quantization**: Use on systems with < 24GB GPU memory
            - **Base size**: Increase for quality, decrease for speed
            - **Prompt types**: 'grounding' for structured docs, 'free ocr' for general text
            - **Batch processing**: Point to a folder to process multiple files at once
            - **Export formats**: Choose between Markdown, HTML, or JSON output
            """)

    return app


if __name__ == '__main__':
    app = create_interface()
    app.launch(
        share=True,
        server_name='127.0.0.1',
        server_port=7860
    )
