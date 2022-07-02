from time import sleep
from threading import Thread, Event
from queue import Queue
from flask import Flask, request, redirect, url_for

from vision import run_vision, game

queue_send = Queue()
queue_receive = Queue()
event = Event()
app = Flask(__name__)


@app.route('/api/game-status')
def game_status():
    if game is not None:
        return str(game)
    return ''


@app.route('/new-game', method='POST')
def start_game():
    if game is None:
        name_list = []
        for field in ['player' + str(i) for i in range(1, 5)]:
            player = request.form[field]
            if player != '':
                name_list.append(player)
        print(name_list)
        run_vision(name_list)
    redirect(url_for('game'))


@app.route('/game')
def game_manager():
    # Buttons: amend score, next player
    return """Game in progress"""


@app.route('/darts')
def main_page():
    return """
<form action="/post" method="post">
  <label for="player1">Player 1:</label>
  <input type="text" id="player1" name="player1"><br><br>
  <label for="player2">Player 2:</label>
  <input type="text" id="player2" name="player2"><br><br>
  <label for="player3">Player 3:</label>
  <input type="text" id="player3" name="player3"><br><br>
  <label for="player4">Player 4:</label>
  <input type="text" id="player4" name="player4"><br><br>
  <input type="submit" value="Start game">
</form> 
"""
