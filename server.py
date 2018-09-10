from flask import Flask
from flask import render_template
from flask import request
import sys
import time
import json
import classify
sys.path.append("./tagger_top_lookup/")
# sys.path.append("./time/")

app = Flask(__name__)

# from evaluate import get_ne
print('============ ner imported ==========')

def ans(query):
    # remain, dic['time'] = parse_year_range(query)
    # dic['time'] = str(dic['time'])
    # dic['intent'] = get_intent(remain)[0]
    start = time.time()
    # uniq_dic, tags, or_tags = get_ne(query)
    file = open('data/input.txt', "w", encoding="utf8")
    file.write('label\tbody\n')
    file.write(str(0) + '\t' + query + '\n')
    file.write(str(0) + '\t' + query + '\n')
    file.close()
    tag = classify.get_sentence()
    uniq_dic = {'intent': str(tag)}
    end = time.time()
    print('[LOG INFO] parse time:', end-start)
    # dic['entities'] =  or_tags
    # dic['documents'], dic['suggest']  = router(dic['intent'], uniq_dic)
    return json.dumps(uniq_dic)

@app.route('/query/<query>')
def QA(query):
    # if request.method == 'POST':
    #     answer = ans(request.form['query'])
        # print ('query --------------------', request.form['query'])
        # return render_template('qa.html', **answer)
    #     return answer
    if request.method == 'GET':
        answer = ans(query)
        return answer
    else:
        return 'hello world'

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5010)
