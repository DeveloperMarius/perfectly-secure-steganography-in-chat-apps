import json

with open('../chat_histories.json', 'r') as f:
    histories = json.load(f)

    for history in histories:
        all_messages = "\n".join([h[10:] for h in history[-20:-1]])
        print(all_messages)

        print("\n\n\n")
