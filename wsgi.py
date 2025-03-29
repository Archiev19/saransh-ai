import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/YOUR_USERNAME/saransh_ai/app.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # Add your project directory to the Python path
    path = '/home/YOUR_USERNAME/saransh_ai'
    if path not in sys.path:
        sys.path.append(path)
        logger.info(f"Added {path} to Python path")

    # Set environment variables
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['TRANSFORMERS_CACHE'] = '/home/YOUR_USERNAME/.cache/huggingface'
    os.environ['TORCH_HOME'] = '/home/YOUR_USERNAME/.cache/torch'
    os.environ['HF_HOME'] = '/home/YOUR_USERNAME/.cache/huggingface'
    
    # Create cache directories if they don't exist
    cache_dirs = [
        '/home/YOUR_USERNAME/.cache/huggingface',
        '/home/YOUR_USERNAME/.cache/torch'
    ]
    for cache_dir in cache_dirs:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")

    # Import your Flask app
    from app import app as application
    logger.info("Successfully imported Flask application")

except Exception as e:
    logger.error(f"Error in WSGI configuration: {str(e)}")
    raise 