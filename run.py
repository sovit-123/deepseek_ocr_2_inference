from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from pdf2image import convert_from_path

import torch
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def load_model(args):
    model_name = 'deepseek-ai/DeepSeek-OCR-2'
    
    quantized_config = BitsAndBytesConfig(
        load_in_4bit=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        _attn_implementation='flash_attention_2', 
        trust_remote_code=True, 
        use_safetensors=True,
        quantization_config=quantized_config if args.int4 else None,
        torch_dtype=torch.bfloat16
    )

    if args.int4:
        model = model.eval()
    else:
        model = model.eval().cuda().to(torch.bfloat16)

    return model, tokenizer

def read_pdf(file_path):
    """
    Read a PDF file and convert each page to an image.
    And save each image in image as a JPEG file.
    The subfolder name is the PDF file name without extension.
    Return a list of image file paths.
    """
    images = convert_from_path(file_path)
    image_files = []
    os.makedirs('images', exist_ok=True)
    # Create file name subdir.
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    image_dir = os.path.join('images', file_name)
    os.makedirs(image_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_file = os.path.join(image_dir, f"page_{i}.jpg")
        image.save(image_file, 'JPEG')
        image_files.append(image_file)

    return image_files

def run_inference(image_files, file_path, model, tokenizer):
    """
    Read images from a list and run inference on each image.
    Save separate makrdowns and then merge into a single markdown as well.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join('outputs', file_name)
    os.makedirs(output_dir, exist_ok=True)

    for i, image_file in enumerate(image_files):
        # if i == 2: break # For testing.
        if args.prompt == 'grounding':
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        else:
            prompt = "<image>\nFree OCR"
        output_path = os.path.join(output_dir, f"page_{i}")
        res = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_file, 
            output_path=output_path, 
            base_size=1024, 
            image_size=768, 
            crop_mode=True, 
            save_results=True
        )
        print(f"Processed {image_file}, results saved to {output_dir}")

    return output_dir

def merge_markdowns(output_dir, merged_markdown):
    # Merge all markdown files into a single markdown file.
    merged_markdown = os.path.join(output_dir, merged_markdown)
    with open(merged_markdown, 'w') as outfile:
        # Read all file with .md extension in output_dir
        for i, dir_name in enumerate(sorted(os.listdir(output_dir))):

            file_name = os.path.join(output_dir, dir_name, 'result.mmd')
            if os.path.exists(file_name):
                with open(file_name, 'r') as infile:
                    content = infile.read()

                    # Find the image placeholders, that is "![]" and replace the path inside the ().
                    content = content.replace("![](images/", f"![]({dir_name}/images/")
                    outfile.write(content + "\n\n")

    print(f"Merged markdown saved to {merged_markdown}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DeepSeek OCR inference on a PDF')
    parser.add_argument('pdf', help='Path to input PDF')
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

    args = parser.parse_args()

    model, tokenizer = load_model(args)

    image_files = read_pdf(args.pdf)

    output_dir = run_inference(
        image_files, 
        file_path=args.pdf, 
        model=model, 
        tokenizer=tokenizer
    )

    merge_markdowns(output_dir, merged_markdown=args.merged)