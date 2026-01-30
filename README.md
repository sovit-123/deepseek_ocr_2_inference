# DeepSeek OCR 2 Document Processing

A document OCR pipeline that converts PDF files to markdown using the DeepSeek OCR 2 model. The system extracts text and structure from document images and outputs structured markdown files.

## Features

- **PDF to Image Conversion**: Automatically converts PDF pages to JPEG images
- **Advanced OCR**: Uses DeepSeek OCR 2 model with 4-bit quantization for efficient processing
- **Markdown Output**: Converts document images to structured markdown format
- **Multi-Page Processing**: Processes multiple pages and merges results into a single markdown file

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support
- Poppler (for PDF to image conversion)

## Installation

### Clone the Repository
```bash
git clone https://github.com/sovit-123/deepseek_ocr_2_inference.git
cd deepseek_ocr_2_inference
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Edit the `file_path` variable in `run.py` with your PDF file path and run:

```bash
python run.py
```

### Configuration

Key parameters in `run.py`:

- `model_name`: Model to use (default: `'deepseek-ai/DeepSeek-OCR-2'`)
- `CUDA_VISIBLE_DEVICES`: GPU device ID (default: `'0'`)
- `base_size`: Image base size for processing (default: `1024`)
- `image_size`: Target image size (default: `768`)
- `crop_mode`: Enable crop mode for better accuracy (default: `True`)

### Output Structure

```
outputs/
├── <document_name>/
│   ├── merged_output.md          # Combined markdown of all pages
│   ├── page_0/
│   │   ├── result.mmd           # Page 0 markdown
│   │   └── images/              # Page 0 extracted images
│   ├── page_1/
│   │   ├── result.mmd
│   │   └── images/
│   └── ...
```

## Project Structure

```
.
├── run.py                    # Main script
├── requirements.txt          # Python dependencies
├── .gitignore              # Git ignore rules
├── images/                 # Input images directory
├── outputs/                # Output results directory
└── input/                  # Input directory
```

## How It Works

1. **PDF Conversion**: Converts each page of the PDF to a JPEG image
2. **Inference**: Runs DeepSeek OCR 2 model on each image with quantization
3. **Markdown Generation**: Converts document content to markdown format
4. **Merging**: Combines individual page markdowns into a single output file with correct image path references

## Performance Notes

- Model is loaded with 4-bit quantization to reduce memory usage
- Flash Attention 2 is used for faster inference
- GPU memory usage: ~6GB (with quantization)
- Processing speed: ~30 seconds -1 minute per page depending on document complexity

## License

Please check the license of the DeepSeek model and included dependencies.

## References

- [DeepSeek OCR 2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
