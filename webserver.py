from bottle import get, post, run, request
from time import sleep
from threading import Thread, Event
from queue import Queue


def do_sleep():
    for i in range(60):
        sleep(1)
    return


@get('/button')
def fn():
    return """
<form action="/button" method="post">
  <label for="player1">Player 1:</label>
  <input type="text" id="player1" name="player1"><br><br>
  <label for="player2">Player 2:</label>
  <input type="text" id="player2" name="player2"><br><br>
  <input type="submit" value="Submit">
</form> 
"""


@post('/button')
def do_print():
    player_1 = request.forms.get('player1')
    player_2 = request.forms.get('player2')
    print(f'Player 1: {player_1}\nPlayer 2: {player_2}')
    do_sleep()
    # return '<p>Sent</p>'
    return """"""


@post('/post')
def check_multitask():
    return 'Multitask!'


@get('/darts')
def main_page():
    return """
<form action="/post" method="post">
  <label for="players">Player:</label>
  <input type="text" id="players" name="players"><br><br>
  <input type="submit" value="Start game">
</form> 
"""


def main():
    queue_send = Queue()
    queue_receive = Queue()
    event = Event()
    run(host='192.168.8.172', port=8080)


if __name__ == '__main__':
    main()
