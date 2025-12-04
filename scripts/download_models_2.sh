cd /home/ugdf8/25Fall/intro-llm-project && python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = '/media/12TB/shared/models'

models = [
    'CohereLabs/aya-expanse-8b',
    'Qwen/Qwen2.5-7B-Instruct',
    'openai/gpt-oss-120b'
]

print('Starting model downloads...')
for model_id in models:
    print(f'\n[{models.index(model_id)+1}/{len(models)}] Downloading {model_id}...')
    trust_remote = 'Qwen' in model_id or 'gpt' in model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote
        )
        print(f'  ✓ Tokenizer downloaded')
        
        # Download model config and weights (not loading into memory)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=HF_TOKEN,
            trust_remote_code=trust_remote
        )
        print(f'  ✓ Config downloaded')
        
        # This will download the model files but not load them
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=HF_TOKEN,
            ignore_patterns=['.msgpack', '.h5', 'optimizer.pt']
        )
        print(f'  ✓ Model weights downloaded')
    except Exception as e:
        print(f'  ✗ Error: {str(e)}')

print('\nAll downloads complete!')
"
