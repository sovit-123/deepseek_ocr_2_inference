from fileinput import filename
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from pdf2image import convert_from_path

import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR-2'

quantoized_config = BitsAndBytesConfig(
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True,
    quantization_config=quantoized_config,
    torch_dtype=torch.bfloat16
)
# model = model.eval().cuda().to(torch.bfloat16)
model = model.eval()

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

def run_inference(image_files, file_path):
    """
    Read images from a list and run inference on each image.
    Save separate makrdowns and then merge into a single markdown as well.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join('outputs', file_name)
    os.makedirs(output_dir, exist_ok=True)

    for i, image_file in enumerate(image_files):
        # if i == 5: break
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        output_path = os.path.join(output_dir, f"page_{i}")
        res = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_file, 
            output_path=output_path, 
            base_size=1024, 
            image_size=768, 
            crop_mode=True, 
            save_results = True
        )
        print(f"Processed {image_file}, results saved to {output_dir}")

    return output_dir

def merge_markdowns(output_dir, merged_markdown):
    # Merge all markdown files into a single markdown file.
    merged_markdown = os.path.join(output_dir, merged_markdown)
    with open(merged_markdown, 'w') as outfile:
        # Read all file with .md extension in output_dir
        for i, dir_name in enumerate(sorted(os.listdir(output_dir))):

            # if os.path.isdir(os.path.join(output_dir, dir_name)) and len(os.listdir(os.path.join(output_dir, dir_name, 'images'))) != 0:
            #     # Create an `images` directory in output_dir if not exists
            #     os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
            #     # Copy the image from the subdir to output_dir/images
            #     for img_file in os.listdir(os.path.join(output_dir, dir_name, 'images')):
            #         src_img_path = os.path.join(output_dir, dir_name, 'images', img_file)
            #         dst_img_path = os.path.join(output_dir, 'images', img_file)
            #         if not os.path.exists(dst_img_path):
            #             os.symlink(src_img_path, dst_img_path)

            file_name = os.path.join(output_dir, dir_name, 'result.mmd')
            if os.path.exists(file_name):
                with open(file_name, 'r') as infile:
                    content = infile.read()

                    # Find the image placeholders, that is "![]" and replace the path inside the ().
                    content = content.replace("![](images/", f"![]({dir_name}/images/")
                    outfile.write(content + "\n\n")

    print(f"Merged markdown saved to {merged_markdown}")

file_path = '/home/sovit/Documents/elon_wiki.pdf'
image_files = read_pdf(file_path)

output_dir = run_inference(image_files, file_path=file_path)

merge_markdowns(output_dir, merged_markdown='merged_output.md')