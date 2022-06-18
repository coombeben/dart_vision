def parse_score(score: str) -> (int, bool):
    if score[0] == 'T':
        return 3 * int(score[1:]), False
    elif score[0] == 'D':
        return 2 * int(score[1:]), True
    elif score == 'Bull':
        return 50, True
    elif score == 'Semibull':
        return 25, False
    return int(score), False


class Game:
    def __init__(self, players, mode='501'):
        if len(players) == 0:
            raise ValueError('Must have at least one player')

        self.mode = mode
        self.playing = len(players)
        self.players = []
        self.game_over = False

        if mode == '501':
            target = 501
        elif mode == '301':
            target = 301
        else:
            raise ValueError(f'mode {mode} is not a valid mode')

        for player_name in players:
            self.players.append(Player(player_name, target))

        print(self)

    def __str__(self):
        return '\n'.join([f'Player: {player.name}\tTarget: {player.target}' for player in self.players])

    def play_round(self):
        for player in self.players:
            player.darts_remaining = 3
            while player.darts_remaining > 0:
                pts = input(f"Enter {player.name}'s score: ")
                player.score(pts)
                if player.target == 0:
                    print(player.name + ' has won!')
                    self.game_over = True
                    return
        print(self)


class Player:
    def __init__(self, name: str, target: int):
        self.name = name
        self.target = target
        self.darts_remaining = 3

    def __str__(self):
        print(f'Player: {self.name}\tTarget: {self.target}')

    def score(self, points: str):
        pts, can_finish = parse_score(points)
        if pts < self.target:
            if self.target - pts == 1:
                self._bust('Cannot have 1 remaining!')
            else:
                self._adjust_target(pts)
        elif pts == self.target:
            if can_finish:
                self._adjust_target(pts)
            else:
                self._bust('Need to finish on a double!')
        else:
            self._bust('Cannot exceed target!')

    def _adjust_target(self, pts: int):
        self.target -= pts
        self.darts_remaining -= 1

    def _bust(self, reason: str):
        print('Bust: ' + reason)
        self.darts_remaining = 0


def main():
    game = Game(['Alice', 'Bob'])
    while not game.game_over:
        game.play_round()


if __name__ == '__main__':
    main()
