#!/usr/bin/env python3
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    # Create models directory
    os.makedirs('/app/models', exist_ok=True)
    os.chdir('/app/models')
    
    try:
        print('📥 Downloading all-mpnet-base-v2 model...')
        model1 = SentenceTransformer('all-mpnet-base-v2')
        model1.save('./all-mpnet-base-v2')
        print('✅ SentenceTransformer model saved to image')
        
        print('📥 Downloading flan-t5-small model...')
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
        model2 = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
        tokenizer.save_pretrained('./flan-t5-small-openai-feedback')
        model2.save_pretrained('./flan-t5-small-openai-feedback')
        print('✅ Flan-T5 model saved to image')
        
        print('🎉 All models are now built into Docker image!')
        
    except Exception as e:
        print(f'❌ Error downloading models: {e}')
        exit(1)

if __name__ == "__main__":
    main()
