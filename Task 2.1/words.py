#Randall Monroe's list from https://xkcd.com/simplewriter/ 
#Downloaded from https://xkcd.com/simplewriter/words.js 
#curl -L https://xkcd.com/simplewriter/words.js -o words.js --output-dir . --fail

import os

def get_thing_explainer_vocab():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    words_path = os.path.join(script_dir, 'words.js')

    with open(words_path, encoding='utf-8') as f:
        return f.read().split('=', 1)[1].strip().rstrip(';').strip('"').split('|')

if __name__ == "__main__":

    words = get_thing_explainer_vocab()
    print('\n'.join(words[:10]))