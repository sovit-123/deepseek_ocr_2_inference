# Configuration defaults for DeepSeek OCR inference

# Model settings
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'
TORCH_DTYPE = 'bfloat16'

# Inference settings
BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True

# Processing settings
MAX_PAGES = None  # None means process all pages
SAVE_RESULTS = True

# Device settings
DEFAULT_DEVICE = 'cuda'
DEFAULT_INT4 = False

# Output settings
MERGED_OUTPUT_NAME = 'merged_output.md'
