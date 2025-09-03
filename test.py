from diuzhev_scratch_llm.tokenizer import BPE

bpe = BPE(31)
phrase = 'Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала.'
bpe.fit(phrase)
encoded = bpe.encode(phrase)
print(phrase)
print("".join([bpe.id2token[tok] for tok in encoded]))