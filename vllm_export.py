from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

app = Flask(__name__)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
)
tokenizer = llm.get_tokenizer()




@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    address = data["data"]
    keys_addres =[i["code"] for i in address]
    final_address = [i["prompt"] for i in address]
    address_data_conversation = []

    for i in  final_address:
        address_data_conversation.append(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': i}],
        tokenize=False))


    outputs = llm.generate(
        address_data_conversation,
        SamplingParams(
            temperature=0.5,
            top_p=0.9,
            max_tokens=150,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
        )
    )

    result = []
    for output in range(len(outputs)):
        try:
            generated_text = outputs[output].outputs[0].text.split("<|end_header_id|>")[1].strip()
            result.append({keys_addres[output]:generated_text})
        except:
            result.append({keys_addres[output]:"error"})

    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    
