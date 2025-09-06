import glob
from diuzhev_scratch_llm import BPE

all_text = []
for file_path in glob.glob('data/PushkinTexts/*.*'):
    file = open(file_path, 'r', encoding='utf8')
    all_text.append(file.read())
    
all_text = '\n\n\n'.join(all_text)

bpe_tokenizer = BPE(2000)
bpe_tokenizer.fit(all_text, show_progress=True)
bpe_tokenizer.save("checkpoints/bpe.dill")