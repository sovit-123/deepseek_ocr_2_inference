from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from pdf2image import convert_from_path

import torch
import os
import argparse
import logging
import time
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model(
    model_name=config.MODEL_NAME, 
    int4=config.DEFAULT_INT4, 
    device=config.DEFAULT_DEVICE
):
    """
    Load DeepSeek OCR model and tokenizer.

    :params model_name: HuggingFace model identifier
    :params int4: Whether to use 4-bit quantization
    :params device: Device to load model on ('cuda' or 'cpu')

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f'Loading model: {model_name} (int4={int4}, device={device})')
    
    quantized_config = BitsAndBytesConfig(
        load_in_4bit=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Try Flash Attention 2, fallback to eager if unavailable
    try:
        model = AutoModel.from_pretrained(
            model_name, 
            _attn_implementation='flash_attention_2', 
            trust_remote_code=True, 
            use_safetensors=True,
            quantization_config=quantized_config if int4 else None,
            torch_dtype=torch.bfloat16
        )
        logger.info('Using Flash Attention 2')
    except Exception as e:
        logger.warning(f'Flash Attention 2 not available: {e}. Falling back to eager attention.')
        model = AutoModel.from_pretrained(
            model_name, 
            _attn_implementation='eager', 
            trust_remote_code=True, 
            use_safetensors=True,
            quantization_config=quantized_config if int4 else None,
            torch_dtype=torch.bfloat16
        )

    model = model.eval()
    if not int4 and device == 'cuda':
        model = model.cuda()
        model = model.to(torch.bfloat16)
    
    logger.info('Model loaded successfully')
    return model, tokenizer

def read_pdf(file_path):
    """
    Read a PDF file and convert each page to an image and save as JPEG files.

    :params file_path: Path to input PDF file

    Returns:
        list: List of image file paths
    """
    images = convert_from_path(file_path)
    image_files = []
    os.makedirs('images', exist_ok=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    image_dir = os.path.join('images', file_name)
    os.makedirs(image_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_file = os.path.join(image_dir, f"page_{i}.jpg")
        image.save(image_file, 'JPEG')
        image_files.append(image_file)

    return image_files

def run_inference(
    image_files, 
    file_path, 
    model, 
    tokenizer, 
    output_dir=None,
    prompt='grounding',
    base_size=config.BASE_SIZE,
    image_size=config.IMAGE_SIZE,
    crop_mode=config.CROP_MODE,
    save_results=config.SAVE_RESULTS,
    max_pages=config.MAX_PAGES,
    progress_callback=None
):
    """
    Run OCR inference on a list of images.

    :params image_files: List of image file paths
    :params file_path: Original input file path for naming output directory
    :params model: Loaded model instance
    :params tokenizer: Loaded tokenizer instance
    :params output_dir: Output directory (if None, created from file_path basename)
    :params prompt: Prompt type ('grounding' or 'free ocr')
    :params base_size: Base size for model inference
    :params image_size: Image size for model inference
    :params crop_mode: Whether to use crop mode
    :params save_results: Whether to save results to disk
    :params max_pages: Maximum pages to process (None for all)
    :params progress_callback: Optional callback function(page_idx, total, page_time, result) for streaming

    Returns:
        tuple: (output_dir, stats_dict) where stats_dict contains processing statistics
    """
    if output_dir is None:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join('outputs', file_name)
    
    os.makedirs(output_dir, exist_ok=True)

    if prompt == 'grounding':
        full_prompt = '<image>\n<|grounding|>Convert the document to markdown. '
    else:
        full_prompt = '<image>\nFree OCR'
    
    num_to_process = len(image_files)
    if max_pages is not None:
        num_to_process = min(max_pages, num_to_process)
    
    # Stats tracking
    stats = {
        'total_pages': num_to_process,
        'page_times': [],
        'total_time': 0,
        'start_time': time.time()
    }
    
    logger.info(f'Starting inference on {num_to_process} image(s)')
    
    for i in range(num_to_process):
        page_start = time.time()
        image_file = image_files[i]
        output_path = os.path.join(output_dir, f"page_{i}.mmd")
        res = model.infer(
            tokenizer, 
            prompt=full_prompt, 
            image_file=image_file, 
            output_path=output_path, 
            base_size=base_size, 
            image_size=image_size, 
            crop_mode=crop_mode, 
            save_results=save_results
        )
        page_time = time.time() - page_start
        stats['page_times'].append(page_time)
        
        logger.info(f'Processed {image_file} ({page_time:.2f}s), results saved to {output_dir}')
        
        # Call progress callback for streaming updates
        if progress_callback:
            progress_callback(i, num_to_process, page_time, res)
    
    stats['total_time'] = time.time() - stats['start_time']
    stats['avg_time_per_page'] = stats['total_time'] / num_to_process if num_to_process > 0 else 0
    
    logger.info(f'Inference complete. Total time: {stats["total_time"]:.2f}s, Avg per page: {stats["avg_time_per_page"]:.2f}s')

    return output_dir, stats

def merge_markdowns(output_dir, merged_markdown=config.MERGED_OUTPUT_NAME):
    """
    Merge all page-level markdown files into a single markdown file.

    :params output_dir: Directory containing page_X subdirectories with result.mmd files
    :params merged_markdown: Name of merged output file or full path

    Returns:
        str: Path to merged markdown file
    """
    if os.path.dirname(merged_markdown) == '':
        merged_markdown = os.path.join(output_dir, merged_markdown)
    
    os.makedirs(os.path.dirname(merged_markdown), exist_ok=True)
    
    with open(merged_markdown, 'w') as outfile:
        for i, dir_name in enumerate(sorted(os.listdir(output_dir))):
            file_name = os.path.join(output_dir, dir_name, 'result.mmd')
            if os.path.exists(file_name):
                with open(file_name, 'r') as infile:
                    content = infile.read()
                    # content = content.replace('![](images/', f"![]({dir_name}/images/")
                    content = content.replace('![](images/', f"![]({os.getcwd()}/{output_dir}/{dir_name}/images/")
                    outfile.write(content + '\n\n')

    logger.info(f'Merged markdown saved to {merged_markdown}')
    return merged_markdown

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DeepSeek OCR inference on a PDF or single image')
    parser.add_argument('pdf', nargs='?', help='Path to input PDF')
    parser.add_argument('--image', '-i', help='Path to input image file')
    parser.add_argument(
        '--prompt',
        default='grounding',
        choices=['grounding', 'free ocr'],
    )
    parser.add_argument(
        '--merged', 
        default='merged_output.md', 
        help='Name of merged markdown output'
    )
    parser.add_argument(
        '--int4',
        action='store_true',
        help='Use int4 quantization'
    )
    parser.add_argument(
        '--max-pages',
        dest='max_pages',
        type=int,
        default=config.MAX_PAGES
    )

    args = parser.parse_args()
    # Validate inputs: accept either a PDF positional argument or a single image via --image
    if args.image and args.pdf:
        parser.error('Provide only one of a PDF positional argument or the --image option')

    if not args.image and not args.pdf:
        parser.error('Provide a PDF path or the --image option')

    model, tokenizer = load_model(int4=args.int4)

    if args.image:
        image_files = [args.image]
        input_path = args.image
    else:
        image_files = read_pdf(args.pdf)
        input_path = args.pdf

    output_dir, stats = run_inference(
        image_files,
        file_path=input_path,
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_pages=args.max_pages
    )

    merge_markdowns(output_dir, merged_markdown=args.merged)