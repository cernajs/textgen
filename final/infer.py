from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('./model')
tokenizer = GPT2Tokenizer.from_pretrained('./model')

def generate_story(title, model, tokenizer, max_length=200):
    input_text = f"TITLE: {title}\nSTORY: "

    input = tokenizer(input_text, return_tensors="pt")
    input_ids      = input["input_ids"]
    attention_mask = input["attention_mask"]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    return generated_text


title = "killer wasp brutally murders entire family"

#time the execution
import time
start = time.time()
story = generate_story(title, model, tokenizer)
end = time.time()
print(f"Time taken: {end - start} seconds")
print(story)
