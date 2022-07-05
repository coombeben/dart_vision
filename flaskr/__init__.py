from flask import Flask, render_template, request, url_for


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/darts')
    def darts():
        return render_template('setup.html')

    @app.route('/start', method='POST')
    def post():
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
    def render_game():
        return render_template('base.html')

    return app
