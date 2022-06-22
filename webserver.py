from bottle import route, run


@route('/hello')
def hello():
    return "Hello world!"


run(host='192.168.8.172', port=8080)
